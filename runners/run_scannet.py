#!/usr/bin/env python
# coding: utf-8
# ## Relative pose estimation on ScanNet

import pdb
import os, sys
import cv2
import numpy as np
from tqdm import tqdm
import torch

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

###########################################
# Hyperparameters to be tuned
###########################################
TH_PIXEL = 1.0
# 0 - 5pt
# 1 - 4line
# 2 - 1vp + 3pt
# 3 - 2vp + 2pt
SOLVER_FLAGS = [True, True, True, True]
RUN_POINT_BASED = [0, 1, 2] # 0 - SuperPoint+SuperGlue; 1 - junctions; 2 - both
RUN_LINE_BASED = []

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
device = 'cuda'
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
pose_errors = []
for data in tqdm(dataloader):
    img1 = data["img1"]
    img2 = data["img2"]
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gt_R_1_2 = data["R_1_2"]
    gt_T_1_2 = data["T_1_2"]
    K1 = data["K1"]
    K2 = data["K2"]

    # Detect, describe and match lines
    line_feat1 = line_matcher.detect_and_describe_lines(gray_img1)
    line_feat2 = line_matcher.detect_and_describe_lines(gray_img2)
    _, m_lines1, m_lines2 = line_matcher.match_lines(gray_img1, gray_img2, line_feat1, line_feat2)

    # Compute and match VPs
    m_lines1_inl = m_lines1[:, :, [1, 0]]
    vp1, label1 = verify_pyprogressivex(gray_img1, m_lines1_inl, threshold=1.5)
    m_lines2_inl = m_lines2[:, :, [1, 0]]
    vp2, label2 = verify_pyprogressivex(gray_img2, m_lines2_inl, threshold=1.5)
    m_vp1, m_label1, m_vp2, m_label2 = vp_matching(vp1, label1, vp2, label2)
    
    # Detect keypoints by SuperPoint + SuperGlue
    mkpts, _ = sg_matching(gray_img1, gray_img2, superglue_matcher, device)
    
    # Run point-based estimators
    for config in RUN_POINT_BASED:
        run_point_based_relative_pose(K1, K2, 
            mkpts,
            mkpts,
            TH_PIXEL,
            config)
           
    # Evaluate the relative pose
    # TODO: compute the relative pose from VP and homography association
    # pred_R_1_2, pred_T_1_2, pts1_inl, pts2_inl = find_relative_pose_from_points(mkpts, K1, K2)
    pred_R_1_2, pred_T_1_2 = run_hybrid_relative_pose(K1, K2,
                                                      [m_lines1_inl.reshape(m_lines1_inl.shape[0], -1).transpose(), m_lines2_inl.reshape(m_lines2_inl.shape[0], -1).transpose()],
                                                      [m_vp1.transpose(), m_vp2.transpose()],
                                                      [mkpts[:,:2].transpose(), mkpts[:,2:4].transpose()],
                                                      [m_label1, m_label2],
                                                      th_pixel=TH_PIXEL,
                                                      solver_flags=SOLVER_FLAGS)
    if pred_R_1_2 is None:
        pose_errors.append(np.inf)
    else:
        pose_errors.append(max(evaluate_R_t(gt_R_1_2, gt_T_1_2, pred_R_1_2, pred_T_1_2)))

auc = 100 * np.r_[pose_auc(pose_errors, thresholds=[5, 10, 20])]
print(f"Median pose error: {np.median(pose_errors):.2f}")
print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")

