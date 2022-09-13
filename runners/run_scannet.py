#!/usr/bin/env python
# coding: utf-8
# ## Relative pose estimation on ScanNet

import os, sys
import cv2
import numpy as np
from tqdm import tqdm
import torch

from robust_line_based_estimator.datasets.scannet import ScanNet
from robust_line_based_estimator.line_matcher import LineMatcher
from robust_line_based_estimator.vp_matcher import vp_matching
from robust_line_based_estimator.evaluation import evaluate_R_t, pose_auc
from robust_line_based_estimator.visualization import (plot_images, plot_lines, plot_color_line_matches,
                                                       plot_vp, plot_keypoints, plot_matches)
from third_party.SuperGluePretrainedNetwork.models.matching import Matching
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import verify_pyprogressivex, sg_matching, find_homography_points, find_relative_pose_from_points

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
device = 'cuda'
superglue_matcher = Matching(config).eval().to(device)

###########################################
# Initialize the line method
###########################################
line_method = 'lsd'  # 'lsd' or 'SOLD2' supported for now
if line_method == 'sold2':
    # SOLD2 matcher
    conf = {
        'sold2': {
            'ckpt_path': '../third_party/SOLD2/pretrained_models/sold2_wireframe.tar',
            'device': 'cpu'
        }
    }
    sold2_matcher = LineMatcher(line_detector='sold2', line_matcher='sold2', conf=conf)
else:
    # LSD+LBD matcher
    lsd_lbd_matcher = LineMatcher(line_detector='lsd', line_matcher='lbd')

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
    if line_method == 'sold2':
        line_feat1 = sold2_matcher.detect_and_describe_lines(gray_img1)
        line_feat2 = sold2_matcher.detect_and_describe_lines(gray_img2)
        _, m_lines1, m_lines2 = sold2_matcher.match_lines(gray_img1, gray_img2, line_feat1, line_feat2)
    else:
        line_feat1 = lsd_lbd_matcher.detect_and_describe_lines(gray_img1)
        line_feat2 = lsd_lbd_matcher.detect_and_describe_lines(gray_img2)
        _, m_lines1, m_lines2 = lsd_lbd_matcher.match_lines(gray_img1, gray_img2, line_feat1, line_feat2)

    # Compute and match VPs
    m_lines1_inl = m_lines1[:, :, [1, 0]]
    vp1, label1 = verify_pyprogressivex(gray_img1, m_lines1_inl, threshold=1.5)
    m_lines2_inl = m_lines2[:, :, [1, 0]]
    vp2, label2 = verify_pyprogressivex(gray_img2, m_lines2_inl, threshold=1.5)
    m_vp1, m_label1, m_vp2, m_label2 = vp_matching(vp1, label1, vp2, label2)

    # Evaluate the relative pose
    # TODO: compute the relative pose from VP and homography association
    mkpts, _ = sg_matching(gray_img1, gray_img2, superglue_matcher, device)
    pred_R_1_2, pred_T_1_2, pts1_inl, pts2_inl = find_relative_pose_from_points(mkpts, K1, K2)
    if pred_R_1_2 is None:
        pose_errors.append(np.inf)
    else:
        pose_errors.append(max(evaluate_R_t(gt_R_1_2, gt_T_1_2, pred_R_1_2, pred_T_1_2)))

auc = 100 * np.r_[pose_auc(pose_errors, thresholds=[5, 10, 20])]
print(f"Median pose error: {np.median(pose_errors):.2f}")
print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")

