#!/usr/bin/env python
# coding: utf-8
# ## Relative pose estimation on ScanNet
import os, sys
import cv2
import numpy as np
from tqdm import tqdm
import torch
import time
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
from robust_line_based_estimator.line_junction_utils import append_h5, read_h5, get_endpoint_correspondences, angular_check
from robust_line_based_estimator.point_based_relative_pose import run_point_based_relative_pose

sys.path.append("build/robust_line_based_estimator")
import line_relative_pose_estimators as _estimators

###########################################
# Hyperparameters to be tuned
###########################################
TH_PIXEL = 3.0
ANGLE_THRESHOLD = math.pi / 16
# 0 - 5pt
# 1 - 4line
# 2 - 1vp + 3pt
# 3 - 2vp + 2pt
SOLVER_FLAGS = [True, False, False, False]
RUN_LINE_BASED = []
USE_ENDPOINTS = False
MAX_JUNCTIONS = 0
OUTPUT_DB_PATH = "scannet_matches.h5"
CORE_NUMBER = 1

###########################################
# Initialize the dataset
###########################################
dataset = ScanNet(root_dir=os.path.expanduser("~/data/ScanNet_relative_pose"), split='test')
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
def process_pair(data, line_matcher, point_matches, CORE_NUMBER, OUTPUT_DB_PATH):
    img1 = data["img1"]
    img2 = data["img2"]
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gt_R_1_2 = data["R_1_2"]
    gt_T_1_2 = data["T_1_2"]
    K1 = data["K1"]
    K2 = data["K2"]
    # Database labels
    label1 = "-".join(data["id1"].split("/")[-3:])
    label2 = "-".join(data["id2"].split("/")[-3:])

    # Try loading the SuperPoint + SuperGlue matches from the database file
    start_time = time.time()
    point_matches = read_h5(f"sp-sg-{label1}-{label2}", OUTPUT_DB_PATH)
    if point_matches is None:
        # Detect keypoints by SuperPoint + SuperGlue
        point_matches, _ = sg_matching(gray_img1, gray_img2, superglue_matcher, device)
        # Saving to the database
        append_h5({f"sp-sg-{label1}-{label2}": point_matches}, OUTPUT_DB_PATH)
    elapsed_time = time.time() - start_time
    if CORE_NUMBER < 2:
        print(f"SP+SG time = {elapsed_time * 1000:.2f} ms")

    # Detect, describe and match lines
    label = f"{line_method}-{matcher_type}-{label1}-{label2}"
    start_time = time.time()
    m_lines1 = read_h5(f"{label}-1", OUTPUT_DB_PATH)
    m_lines2 = read_h5(f"{label}-2", OUTPUT_DB_PATH)

    if m_lines1 is None or m_lines2 is None:
        # Detect lines in the images
        line_feat1 = line_matcher.detect_and_describe_lines(gray_img1)
        line_feat2 = line_matcher.detect_and_describe_lines(gray_img2)
        elapsed_time = time.time() - start_time
        if CORE_NUMBER < 2:
            print(f"Line detection detection time = {elapsed_time * 1000:.2f} ms")

        # Match lines in the images
        start_time = time.time()
        _, m_lines1, m_lines2 = line_matcher.match_lines(gray_img1, gray_img2, line_feat1, line_feat2)
        elapsed_time = time.time() - start_time
        if CORE_NUMBER < 2:
            print(f"Line matching detection time = {elapsed_time * 1000:.2f} ms")

        # Saving to the database
        append_h5({f"{label}-1": m_lines1,
            f"{label}-2": m_lines2}, OUTPUT_DB_PATH)
    else:
        elapsed_time = time.time() - start_time
        if CORE_NUMBER < 2:
            print(f"Line detection/matching detection time = {elapsed_time * 1000:.2f} ms")

    # Run point-based estimation if no line correspondences are found
    if m_lines1.shape[0] < 2:
        start_time = time.time()
        R, t = run_point_based_relative_pose(K1, K2,
            point_matches,
            np.array([]),
            th_pixel = TH_PIXEL,
            config = 0)
        elapsed_time = time.time() - start_time
        return max(evaluate_R_t(gt_R_1_2, gt_T_1_2, R, t)), elapsed_time

    # Compute and match VPs or load them from the database
    m_vp1 = read_h5(f"{label}-vp1", OUTPUT_DB_PATH)
    m_vp2 = read_h5(f"{label}-vp2", OUTPUT_DB_PATH)
    m_label1 = read_h5(f"{label}-vpl1", OUTPUT_DB_PATH)
    m_label2 = read_h5(f"{label}-vpl2", OUTPUT_DB_PATH)
    m_lines1_inl = m_lines1[:, :, [1, 0]]
    m_lines2_inl = m_lines2[:, :, [1, 0]]

    start_time = time.time()
    if m_vp1 is None or m_vp2 is None or m_label1 is None or m_label2 is None:
        # Detect vanishing points in the source image
        vp1, vp_label1 = verify_pyprogressivex(gray_img1, m_lines1_inl, threshold=1.5)
        # Detect vanishing points in the destination image
        vp2, vp_label2 = verify_pyprogressivex(gray_img2, m_lines2_inl, threshold=1.5)
        # Matching the vanishing points
        m_vp1, m_label1, m_vp2, m_label2 = vp_matching(vp1, vp_label1, vp2, vp_label2)

        # Saving to the database
        append_h5({f"{label}-vp1": m_vp1,
            f"{label}-vp2": m_vp2,
            f"{label}-vpl1": m_label1,
            f"{label}-vpl2": m_label2}, OUTPUT_DB_PATH)
        elapsed_time = time.time() - start_time
    if CORE_NUMBER < 2:
        print(f"VP detection time = {elapsed_time * 1000:.2f} ms")

    if np.array(m_vp1).shape[0] == 0:
        start_time = time.time()
        R, t = run_point_based_relative_pose(K1, K2,
            point_matches,
            np.array([]),
            th_pixel = TH_PIXEL,
            config = 0)
        elapsed_time = time.time() - start_time
        return max(evaluate_R_t(gt_R_1_2, gt_T_1_2, R, t)), elapsed_time

    # Adding the line endpoints as point correspondences
    if USE_ENDPOINTS:
        endpoints = get_endpoint_correspondences(m_lines1, m_lines2)
        point_matches = np.concatenate((point_matches, endpoints), axis=0)

    # Evaluate the relative pose
    # [Note] First construct those junction instances!!
    junctions_1, junctions_2 = [], []
    for idx in range(point_matches.shape[0]):
        junctions_1.append(_estimators.Junction2d(point_matches[idx][:2]))
        junctions_2.append(_estimators.Junction2d(point_matches[idx][2:]))

    # Add only those lines that are almost orthogonal
    line_pairs = angular_check(m_lines1, m_lines2, K1, K2, img1, img2)
    for i, j, a1, a2 in line_pairs:
        if a1 > math.pi:
            a1 -= math.pi
        e1 = abs(a1 - math.pi / 2)
        if a2 > math.pi:
            a2 -= math.pi
        e2 = abs(a2 - math.pi / 2)
        e = max(e1, e2)
        if e < ANGLE_THRESHOLD:
            line11 = np.reshape(m_lines1[i], (4, 1))
            line12 = np.reshape(m_lines1[j], (4, 1))
            line21 = np.reshape(m_lines2[i], (4, 1))
            line22 = np.reshape(m_lines2[j], (4, 1))

            junctions_1.append(_estimators.Junction2d(line11, line12))
            junctions_2.append(_estimators.Junction2d(line21, line22))

            if len(junctions_1) >= point_matches.shape[0] + MAX_JUNCTIONS:
                break

    start_time = time.time()
    pred_R_1_2, pred_T_1_2 = run_hybrid_relative_pose(K1, K2,
                                                      [m_lines1_inl.reshape(m_lines1_inl.shape[0], -1).transpose(), m_lines2_inl.reshape(m_lines2_inl.shape[0], -1).transpose()],
                                                      [m_vp1.transpose(), m_vp2.transpose()],
                                                      [junctions_1, junctions_2],
                                                      [m_label1, m_label2],
                                                      th_pixel=TH_PIXEL,
                                                      solver_flags=SOLVER_FLAGS)
    elapsed_time = time.time() - start_time
    if CORE_NUMBER < 2:
        print(f"Estimation time = {elapsed_time * 1000:.2f} ms")
    return max(evaluate_R_t(gt_R_1_2, gt_T_1_2, pred_R_1_2, pred_T_1_2)), elapsed_time

print("Collecting data...")
processing_queue = []
for data in tqdm(dataloader):
    processing_queue.append(data)
    #if len(processing_queue) >= 50:
    #    break

print("Running estimators...")
results = Parallel(n_jobs=min(CORE_NUMBER, len(processing_queue)))(delayed(process_pair)(
    data,
    line_matcher,
    superglue_matcher,
    CORE_NUMBER,
    OUTPUT_DB_PATH) for data in tqdm(processing_queue))

pose_errors = [error for error, time in results]
runtimes = [time for error, time in results]

pose_errors = np.array(pose_errors)

auc = 100 * np.r_[pose_auc(pose_errors, thresholds=[5, 10, 20])]
print(f"Average run-time: {1000 * np.mean(runtimes):.2f} ms")
print(f"Median pose error: {np.median(pose_errors):.2f}")
print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")

