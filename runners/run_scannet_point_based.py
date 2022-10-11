#!/usr/bin/env python
# coding: utf-8
# ## Relative pose estimation on ScanNet

import pdb
import os, sys
from pickle import TRUE
import cv2
import numpy as np
from tqdm import tqdm
import torch
import math
from joblib import Parallel, delayed

from robust_line_based_estimator.datasets.scannet import ScanNet
from robust_line_based_estimator.line_matching.line_matcher import LineMatcher
from robust_line_based_estimator.vp_matcher import vp_matching
from robust_line_based_estimator.evaluation import evaluate_R_t, pose_auc
from robust_line_based_estimator.visualization import (plot_images, plot_lines, plot_color_line_matches,
                                                       plot_vp, plot_keypoints, plot_matches)
from third_party.SuperGluePretrainedNetwork.models.matching import Matching
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import verify_pyprogressivex, sg_matching, find_homography_points, find_relative_pose_from_points
from robust_line_based_estimator.hybrid_relative_pose import run_hybrid_relative_pose
from robust_line_based_estimator.point_based_relative_pose import run_point_based_relative_pose
from robust_line_based_estimator.line_junction_utils import append_h5, read_h5, get_line_junctions, filter_points_by_relative_pose, depth_consistency_check, get_endpoint_correspondences, angular_check

###########################################
# Hyperparameters to be tuned
###########################################
TH_PIXEL = 1.5
THR_PLANARITY = 0.001 #5
RUN_POINT_BASED = [0, 2]#, 1, 2] # 0 - SuperPoint+SuperGlue; 1 - junctions; 2 - both
RUN_LINE_BASED = []
OUTPUT_DB_PATH = "scannet_matches.h5" 
DEPTH_PATH = "/media/hdd3tb/datasets/scannet/scannet_test_images"
USE_ENDPOINT_INSTEAD_OF_JUNCTIONS = False
FILTER_JUNCTIONS_BY_POSE = False
REJECT_JUNCTIONS_OUTSIDE = False
FILTER_JUNCTIONS_BY_DEPTH = True
ORDER_BY_ANGLE = True
CORE_NUMBER = 18

###########################################
# Initialize the dataset
###########################################
dataset = ScanNet(root_dir=os.path.expanduser("/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test"), split='test')
dataloader = dataset.get_dataloader()

###########################################
# Initialize SuperPoint + SuperGlue (only used as a point baseline)
###########################################
config = {
    'superpoint': {
        'nms_radius': 4,
        'max_keypoints': 1024,
    },
    'superglue': {'weights': 'indoor'}
}
device = 'cpu'
superglue_matcher = Matching(config).eval().to(device)

###########################################
# Initialize the line method
###########################################
line_method = 'lsd'  # 'lsd' or 'SOLD2' supported for now
matcher_type  = "superglue_endpoints"
if matcher_type == 'sold2':
    # SOLD2 matcher
    conf = {
        'sold2': {
            'ckpt_path': '../third_party/SOLD2/pretrained_models/sold2_wireframe.tar',
            'device': 'cpu'
        }
    }
    line_matcher = LineMatcher(line_detector='sold2', line_matcher='sold2', conf=conf)
elif matcher_type == "lbd":
    # LSD+LBD matcher
    line_matcher = LineMatcher(line_detector='lsd', line_matcher='lbd')
elif matcher_type == "superglue_endpoints":
    # SuperGlue matcher
    conf = {
        'sg_params': {
            'weights': 'indoor'
        } 
    }
    line_matcher = LineMatcher(line_detector=line_method, line_matcher='superglue_endpoints', conf=conf)

###########################################
# Relative pose estimation
###########################################
def process_pair(data, line_matcher, point_matches, OUTPUT_DB_PATH):
    img1 = data["img1"]
    img2 = data["img2"]
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gt_R_1_2 = data["R_1_2"]
    gt_T_1_2 = data["T_1_2"]
    K1 = data["K1"]
    K2 = data["K2"]

    # Try loading the SuperPoint + SuperGlue matches from the database file
    label1 = "-".join(data["id1"].split("/")[-3:])
    label2 = "-".join(data["id2"].split("/")[-3:])
    point_matches = read_h5(f"sp-sg-{label1}-{label2}", OUTPUT_DB_PATH)
    if point_matches is None:
        # Detect keypoints by SuperPoint + SuperGlue
        point_matches, _ = sg_matching(gray_img1, gray_img2, point_matches, device)
        # Saving to the database
        append_h5({f"sp-sg-{label1}-{label2}": point_matches}, OUTPUT_DB_PATH)       

    # Try loading the line matches from the database file
    label = f"{line_method}-{matcher_type}-{label1}-{label2}"
    m_lines1 = read_h5(f"{label}-1", OUTPUT_DB_PATH)
    m_lines2 = read_h5(f"{label}-2", OUTPUT_DB_PATH)
    junctions = read_h5(f"{label}-j", OUTPUT_DB_PATH)
    if m_lines1 is None or m_lines2 is None or junctions is None:
        # Detect, describe and match lines
        line_feat1 = line_matcher.detect_and_describe_lines(gray_img1)
        line_feat2 = line_matcher.detect_and_describe_lines(gray_img2)
        _, m_lines1, m_lines2 = line_matcher.match_lines(gray_img1, gray_img2, line_feat1, line_feat2)
        # Calculating the line junctions
        junctions1, junctions2 = get_line_junctions(m_lines1, m_lines2)
        try:
            junctions1 = junctions1[:, [1, 0]]
            junctions2 = junctions2[:, [1, 0]]
            junctions = np.concatenate((junctions1, junctions2), axis=1)
        except:
            junctions = np.array([])
        # Saving to the database
        append_h5({f"{label}-1": m_lines1,
            f"{label}-2": m_lines2,
            f"{label}-j": junctions}, OUTPUT_DB_PATH)       

    # Using the endpoints of the lines as correspondences
    if USE_ENDPOINT_INSTEAD_OF_JUNCTIONS:
        junctions = get_endpoint_correspondences(m_lines1, m_lines2)
        
    # Checking the angle between the line pairs
    valid_line_pairs = None
    if ORDER_BY_ANGLE:
        # Angle calculation
        line_pairs = angular_check(m_lines1, m_lines2, K1, K2, img1, img2)
        # Calculating error w.r.t the lines being orthogonal
        errors = []
        valid_line_pairs = []
        for i, j, a1, a2 in line_pairs:
            a1 = 180.0 * a1 / math.pi
            if a1 > 180.0:
                a1 -= 180.0
            e1 = abs(a1 - 90)
            a2 = 180.0 * a2 / math.pi
            if a2 > 180.0:
                a2 -= 180.0
            e2 = abs(a2 - 90)  
            e = 0.5 * (e1 + e2)
            if e < 10:
                errors.append(e)    
                valid_line_pairs.append((i, j, e))
            
        # Calculating the line junctions from the kept pairs
        junctions1, junctions2, distances = get_line_junctions(m_lines1, m_lines2, valid_line_pairs)
        
        if junctions1.shape[0] > 0:
            junctions1 = junctions1[:, [1, 0]]
            junctions2 = junctions2[:, [1, 0]]
            junctions = np.concatenate((junctions1, junctions2), axis=1)
            
            sorted_indices = np.argsort(distances)
            junctions = junctions[sorted_indices, :]
            distances = [distances[i] for i in sorted_indices]

    if FILTER_JUNCTIONS_BY_DEPTH and not USE_ENDPOINT_INSTEAD_OF_JUNCTIONS:
        depth_img1_path = "_".join(data["id1"].split("/")[-1:-4:-2][::-1]).split(".")[0]
        depth_img2_path = "_".join(data["id2"].split("/")[-1:-4:-2][::-1]).split(".")[0]
        
        # Loading the depth images
        depth_img1 = cv2.imread(f"{DEPTH_PATH}/{depth_img1_path}.depth.png", 0)
        depth_img2 = cv2.imread(f"{DEPTH_PATH}/{depth_img2_path}.depth.png", 0)
        
        # Selecting the line pairs that are coplanar
        depth_valid_line_pairs = depth_consistency_check(m_lines1, m_lines2, K1, K2, img1, img2, depth_img1, depth_img2, THR_PLANARITY, valid_line_pairs)
        #distances = [d for i,j,d in valid_line_pairs]

        # Calculating the line junctions from the kept pairs
        junctions1, junctions2, distances = get_line_junctions(m_lines1, m_lines2, depth_valid_line_pairs)

        if junctions1.shape[0] > 0:
            junctions1 = junctions1[:, [1, 0]]
            junctions2 = junctions2[:, [1, 0]]
            junctions = np.concatenate((junctions1, junctions2), axis=1)

            # Sort by the distances
            sorted_indices = np.argsort(distances)
            junctions = junctions[sorted_indices, :]
            distances = [distances[i] for i in sorted_indices]

    # Filter if the point falls outside the image
    if REJECT_JUNCTIONS_OUTSIDE and len(junctions) > 0:
        mask = (junctions[:, 0] >= 0) & (junctions[:, 0] < img1.shape[1]) & \
            (junctions[:, 1] >= 0) & (junctions[:, 1] < img1.shape[0]) & \
            (junctions[:, 2] >= 0) & (junctions[:, 2] < img2.shape[1]) & \
            (junctions[:, 3] >= 0) & (junctions[:, 3] < img2.shape[0]) 
        if mask.sum() > 5:
            junctions = junctions[mask, :]

    # Filtering the junctions by the ground truth pose
    inlier_ratio = 0.0
    if len(junctions) > 0:
        gt_mask, _ = filter_points_by_relative_pose(K1, K2, gt_R_1_2, gt_T_1_2, junctions[:, :2], junctions[:, 2:], TH_PIXEL)
        inlier_ratio = gt_mask.sum() / junctions.shape[0]
        if FILTER_JUNCTIONS_BY_POSE:
            junctions = junctions[gt_mask,:]

    # Run point-based estimators
    errors = []
    for config in RUN_POINT_BASED:
        R, t = run_point_based_relative_pose(K1, K2, 
            point_matches,
            junctions,
            th_pixel = TH_PIXEL,
            best_k = 5000,
            config = config)

        if R is None or t is None:
            error = np.inf
        else:
            error = max(evaluate_R_t(gt_R_1_2, gt_T_1_2, R, t))
        errors.append(error)
    return errors, inlier_ratio

    pose_errors2 = np.array(pose_errors)
    print(f"Average inlier ratio: {np.mean(inlier_ratios):.2f}")
    for col in range(pose_errors2.shape[1]):
        auc = 100 * np.r_[pose_auc(pose_errors2[:,col], thresholds=[5, 10, 20])]
        print(f"Median pose error: {np.median(pose_errors2[:,col]):.2f}")
        print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")

print("Collecting data...")
processing_queue = []
for data in tqdm(dataloader):
    processing_queue.append(data)

results = Parallel(n_jobs=min(CORE_NUMBER, len(processing_queue)))(delayed(process_pair)(
    data,
    line_matcher,
    superglue_matcher,
    OUTPUT_DB_PATH) for data in tqdm(processing_queue))

pose_errors = [err for err, inl in results]
inlier_ratios = [inl for err, inl in results]

pose_errors = np.array(pose_errors)

print(f"Average inlier ratio: {np.mean(inlier_ratios):.2f}")
for col in range(pose_errors.shape[1]):
    auc = 100 * np.r_[pose_auc(pose_errors[:,col], thresholds=[5, 10, 20])]
    print(f"Median pose error: {np.median(pose_errors[:,col]):.2f}")
    print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")

