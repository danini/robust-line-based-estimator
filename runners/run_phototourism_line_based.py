#!/usr/bin/env python
# coding: utf-8
# ## Relative pose estimation on ScanNet
import os, sys
import cv2
import numpy as np
import ast
from tqdm import tqdm
import time
import math
import csv
from joblib import Parallel, delayed

from robust_line_based_estimator.datasets.phototourism import PhotoTourism
from robust_line_based_estimator.line_matching.line_matcher import LineMatcher
from robust_line_based_estimator.line_matching.gluestick import GlueStick
from robust_line_based_estimator.vp_matcher import vp_matching
from robust_line_based_estimator.evaluation import evaluate_R_t, pose_auc
from third_party.SuperGluePretrainedNetwork.models.matching import Matching
from kornia.feature import LoFTR
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import verify_pyprogressivex, point_matching, joint_vp_detection_and_matching
from robust_line_based_estimator.hybrid_relative_pose import run_hybrid_relative_pose
from robust_line_based_estimator.line_junction_utils import append_h5, read_h5, get_endpoint_correspondences, angular_check, get_actual_junctions
from robust_line_based_estimator.point_based_relative_pose import run_point_based_relative_pose

sys.path.append("build/robust_line_based_estimator")
import line_relative_pose_estimators as _estimators

###########################################
# Hyperparameters to be tuned
###########################################
ANGLE_THRESHOLD = math.pi / 32
# scoring
TH_VP_ANGLE = 1.5
TH_PIXEL = 0.75
DATA_WEIGHTS = [0.0, 0.0, 1.0]
# LO refinement
LS_REFINEMENT = 1 # 0 for sampson, 1 for sampson + vp + line, 2 for sampson + vp (fixed)
WEIGHTS_REFINEMENT = [1000.0, 50.0] # 0 for vp rotation error, 1 for line-vp error
# 0 - 5pt
# 1 - 4pt homography
# 2 - 4line homography
# 3 - 1vp + 3pt
# 4 - 1vp + 3cll
# 5 - 2vp + 2pt
# 6 - 1line + 1vp + 2pt + orthogonal
# 7 - 1vp + 2line + 1pt + orthogonal
SOLVER_FLAGS = [True, False, False, False, False, False, False, False]
RUN_LINE_BASED = []
USE_ENDPOINTS = False
MAX_JUNCTIONS = 0
USE_JOINT_VP_MATCHING = True
REFINE_VP = True
MATCHER = "SG"  # "SG", "LoFTR", or "GS"
matcher_type  = 'gluestick'  # 'lbd', 'sold2', 'superglue_endpoints', or 'gluestick'
line_method = 'deeplsd'  # 'lsd' or 'SOLD2' supported for now
CORE_NUMBER = 19
BATCH_SIZE = 2000
LINE_INLIER_RATIO = 0.3
LINE_CUT_LENGTH = 1
FILTER_THRESHOLD = 2
BLUR_SIGMA = 0.2
SG_THR = 0.3
OUTPUT_DB_PATH = f"matches/phototourism_matches_{MATCHER}_{matcher_type}_outdoor_{line_method}_1_{FILTER_THRESHOLD}_{SG_THR}_{BLUR_SIGMA}_.h5"

SOLVERSs = [
            # SOLVER_FLAGS, TH_VP_ANGLE, TH_PIXEL, MAX_JUNCTIONS, ANGLE_THRESHOLD, USE_JOINT_VP_MATCHING, USE_ENDPOINTS, USE_JUNCTIONS, REFINE_VP, LS_REFINEMENT, LINE_INLIER_RATIO
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[True, False, False, False, False, False, False, False], 0.0, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 0.1, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 0.1, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 0.25, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 0.25, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 0.5, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 0.5, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 1.0, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 1.0, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 1.5, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 1.5, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 2.0, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[False, False, False, True, True, True, True, True], 2.0, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, False, False, False, False, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, True, False, False, False, 0, 0.0],
            
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, False, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, True, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, False, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, True, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, False, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, True, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, False, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, True, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, False, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, True, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, False, False, False, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, True, False, False, True, 0, 0.0],
            
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, False, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, True, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, False, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, True, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, False, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, True, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, False, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, True, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, False, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, True, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, False, False, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, True, False, True, True, 0, 0.0],
            
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, False, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.1, 3.0, 0, 15, True, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, False, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.25, 3.0, 0, 15, True, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, False, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 0.5, 3.0, 0, 15, True, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, False, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.0, 3.0, 0, 15, True, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, False, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 1.5, 3.0, 0, 15, True, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, False, True, True, True, 0, 0.0],
            [[True, True, True, True, True, True, True, True], 2.0, 3.0, 0, 15, True, True, True, True, 0, 0.0],
            
            #[[True, False, False, False, False, False, False, False], 0.0, 2.0, 0, 15, False, True, True, False, 0, 0.0],
            #[[True, False, False, False, False, False, False, False], 0.0, 2.0, 0, 15, False, False, False, False, 0, 0.0],
            #[[True, False, False, False, False, False, False, False], 0.0, 2.0, 0, 15, False, True, False, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, False, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, True, True, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 0, 15, True, True, True, True, 2, 0.0],
            #
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, True, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, False, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, True, True, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 0, 15, True, True, True, True, 2, 0.0],
            #
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 0, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 0, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 0, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 0, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 0, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 0, 15, True, True, True, True, 2, 0.0],
            #
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, False, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, True, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, True, True, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 100, 15, True, True, True, True, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 100, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 100, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 100, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 100, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 100, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 100, 15, True, True, True, True, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 100, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 100, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 100, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 100, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 100, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 100, 15, True, True, True, True, 2, 0.0],
            #
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, False, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, True, False, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, True, True, 0, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.0, 2.0, 200, 15, True, True, True, True, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 200, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 200, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 200, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 200, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 200, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 0.5, 2.0, 200, 15, True, True, True, True, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 200, 15, True, True, False, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 200, 15, True, True, True, False, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 200, 15, True, True, True, True, 1, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 200, 15, True, True, False, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 200, 15, True, True, True, False, 2, 0.0],
            #[[True, True, True, True, True, True, True, True], 1.5, 2.0, 200, 15, True, True, True, True, 2, 0.0],
        ]

###########################################
# Initialize the dataset
###########################################
dataset = PhotoTourism(
    root_dir = os.path.expanduser("/media/hdd2tb/datasets/RANSAC-Tutorial-Data"),
    split='val')
dataloader = dataset.get_dataloader()

###########################################
# Initialize the point matcher
###########################################
device = 'cuda'
if MATCHER == "SG":
    config = {
        'superpoint': {
            'nms_radius': 4,
            'max_keypoints': 1024,
        },
        'superglue': {'weights': 'outdoor'}
    }
    matcher = Matching(config).eval().to(device)
    matcher_key = 'sp-sg'
elif MATCHER == "LoFTR":
    matcher = LoFTR(pretrained='outdoor').to(device)
    matcher_key = 'loftr'
elif MATCHER == "GS":
    config = {    
        'device': device,
        'gs_params': {
            'weights': 'outdoor'
        }
    }
    matcher = GlueStick(config)
    matcher_key = 'gs'
else:
    raise ValueError("Unknown matcher " + MATCHER)

###########################################
# Initialize the line method
###########################################
if matcher_type == 'sold2':
    # SOLD2 matcher
    conf = {
        'sold2': {
            'ckpt_path': '../third_party/SOLD2/pretrained_models/sold2_wireframe.tar',
            'device': 'cuda'
        }
    }
    line_matcher = LineMatcher(line_detector='sold2', line_matcher='sold2',
                               conf=conf)
elif matcher_type == "lbd":
    # LSD+LBD matcher
    line_matcher = LineMatcher(line_detector='lsd', line_matcher='lbd')
elif matcher_type == "superglue_endpoints":
    # SuperGlue matcher
    conf = {
        'sg_params': {
            'weights': 'outdoor',
            'device': 'cuda',
            'match_threshold': SG_THR,
        },
        'sigma_scale': BLUR_SIGMA,
    }
    line_matcher = LineMatcher(line_detector=line_method,
                               line_matcher='superglue_endpoints', conf=conf)
elif matcher_type == "gluestick":
    # GlueStick matcher
    conf = {}
    line_matcher = LineMatcher(line_detector=line_method,
                               line_matcher='gluestick', conf=conf)

###########################################
# Detecting everything before the pose estimation starts
###########################################
def detect_and_load_data(data, line_matcher, CORE_NUMBER):
    img1 = data["img1"]
    img2 = data["img2"]
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Database labels
    label1 = "-".join(data["id1"].split("/")[-3:])
    label2 = "-".join(data["id2"].split("/")[-3:])

    # Try loading the point matches from the database file
    start_time = time.time()
    point_matches = read_h5(f"{matcher_key}-{label1}-{label2}", OUTPUT_DB_PATH)
    if point_matches is None:
        # Detect keypoints by SuperPoint + SuperGlue, LoFTR, or GlueStick
        point_matches, _ = point_matching(gray_img1, gray_img2, MATCHER,
                                          matcher, device)
        # Saving to the database
        append_h5({f"{matcher_key}-{label1}-{label2}": point_matches}, OUTPUT_DB_PATH)
    elapsed_time = time.time() - start_time
    if CORE_NUMBER < 2:
        print(f"{matcher_key} time = {elapsed_time * 1000:.2f} ms")

    # Detect, describe and match lines
    label = f"{line_method}-{matcher_type}-{label1}-{label2}"
    start_time = time.time()
    m_lines1 = read_h5(f"{label}-1", OUTPUT_DB_PATH)
    m_lines2 = read_h5(f"{label}-2", OUTPUT_DB_PATH)
    
    junc1 = read_h5(f"{label}-junc1", OUTPUT_DB_PATH)
    junc2 = read_h5(f"{label}-junc2", OUTPUT_DB_PATH)
    junc_indices = read_h5(f"{label}-junc-indices", OUTPUT_DB_PATH)
    line_pairs = read_h5(f"{label}-line-pairs", OUTPUT_DB_PATH)
    
    if m_lines1 is None or m_lines2 is None:
        # Detect lines in the images
        line_feat1 = line_matcher.detect_and_describe_lines(gray_img1)
        line_feat2 = line_matcher.detect_and_describe_lines(gray_img2)
        elapsed_time = time.time() - start_time
        print(f"Line detection detection time = {elapsed_time * 1000:.2f} ms")

        # Match lines in the images
        start_time = time.time()
        _, m_lines1, m_lines2 = line_matcher.match_lines(gray_img1, gray_img2, line_feat1, line_feat2)
        elapsed_time = time.time() - start_time
        print(f"Line matching detection time = {elapsed_time * 1000:.2f} ms")
        
        # Saving to the database
        append_h5({f"{label}-1": m_lines1,
            f"{label}-2": m_lines2}, OUTPUT_DB_PATH)
        
        start_time = time.time()
        K1 = data["K1"]
        K2 = data["K2"]
        m_lines1_inl = m_lines1[:, :, [1, 0]]
        m_lines2_inl = m_lines2[:, :, [1, 0]]
        junc1, junc2, junc_indices = get_actual_junctions(m_lines1_inl, m_lines2_inl, K1, K2, filter_threshold=FILTER_THRESHOLD)
        junc_indices = np.array(junc_indices)
        elapsed_time = time.time() - start_time
        print(f"Junction finding time = {elapsed_time * 1000:.2f} ms")
        
        start_time = time.time()
        line_pairs = np.array(angular_check(m_lines1, m_lines2, K1, K2))
        elapsed_time = time.time() - start_time
        print(f"Line pair forming time = {elapsed_time * 1000:.2f} ms")
        
        append_h5({f"{label}-junc1": junc1,
            f"{label}-junc2": junc2,
            f"{label}-junc-indices": junc_indices,
            f"{label}-line-pairs": line_pairs}, OUTPUT_DB_PATH)
    else:
        elapsed_time = time.time() - start_time
        #print(f"Line detection/matching detection time = {elapsed_time * 1000:.2f} ms")
    #
    return point_matches, m_lines1, m_lines2, junc1, junc2, junc_indices, line_pairs

###########################################
# Relative pose estimation
###########################################
def process_pair(data, point_matches, m_lines1, m_lines2, junc1, junc2, junc_indices, line_pairs, CORE_NUMBER, SOLVER_FLAGS, TH_PIXEL, USE_ENDPOINTS, USE_JUNCTIONS, REFINE_VP, ANGLE_THRESHOLD, USE_JOINT_VP_MATCHING, TH_VP_ANGLE, LS_REFINEMENT, LINE_INLIER_RATIO):
    gt_R_1_2 = data["R_1_2"]
    gt_T_1_2 = data["T_1_2"]
    K1 = data["K1"]
    K2 = data["K2"]

    # Run point-based estimation if no line correspondences are found
    #if m_lines1.shape[0] < 2:
    #    start_time = time.time()
    #    R, t = run_point_based_relative_pose(K1, K2,
    #        point_matches,
    #        np.array([]),
    #        th_pixel = TH_PIXEL,
    #        config = 0)
    #    elapsed_time = time.time() - start_time
    #    return evaluate_R_t(gt_R_1_2, gt_T_1_2, R, t), elapsed_time

    # Compute and match VPs or load them from the database
    m_lines1_inl = m_lines1[:, :, [1, 0]]
    m_lines2_inl = m_lines2[:, :, [1, 0]]
    m_vp1 = []
    m_vp2 = []
    m_label1 = np.zeros((0))
    m_label2 = np.zeros((0))

    vp_thr = TH_VP_ANGLE
    TH_VP_ANGLE = 0
    
    start_time = time.time()
    if m_lines1.shape[0] >= 2:
        if USE_JOINT_VP_MATCHING:
            # Jointly detect VP and match them
            m_vp1, m_vp2, m_label1 = joint_vp_detection_and_matching(
                int(2 * K1[0,2]), int(2 * K1[1,2]), m_lines1_inl,
                m_lines2_inl, threshold=vp_thr)
            m_label2 = m_label1
        else:
            # Detect vanishing points in the source image
            vp1, vp_label1 = verify_pyprogressivex(
                int(2 * K1[0,2]), int(2 * K1[1,2]),
                m_lines1_inl, threshold=vp_thr)
            # Detect vanishing points in the destination image
            vp2, vp_label2 = verify_pyprogressivex(
                int(2 * K2[0,2]), int(2 * K2[1,2]),
                m_lines2_inl, threshold=vp_thr)
            # Matching the vanishing points
            m_vp1, m_label1, m_vp2, m_label2 = vp_matching(vp1, vp_label1,
                                                        vp2, vp_label2)
            
        if REFINE_VP:
            m_vp1 = _estimators.refine_vp(
                m_label1, m_lines1_inl.reshape(-1, 4, 1), m_vp1)
            m_vp1 = np.array(m_vp1)
            m_vp2 = _estimators.refine_vp(
                m_label2, m_lines2_inl.reshape(-1, 4, 1), m_vp2)
            m_vp2 = np.array(m_vp2)
        elapsed_time = time.time() - start_time
        if CORE_NUMBER < 2:
            print(f"VP detection time = {elapsed_time * 1000:.2f} ms")

    #if np.array(m_vp1).shape[0] == 0:
    #    start_time = time.time()
    #    R, t = run_point_based_relative_pose(K1, K2,
    #        point_matches,
    #        np.array([]),
    #        th_pixel = TH_PIXEL,
    #        config = 0)
    #    elapsed_time = time.time() - start_time
    #    return evaluate_R_t(gt_R_1_2, gt_T_1_2, R, t), elapsed_time

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
    if MAX_JUNCTIONS > 0 and line_pairs is not None:
        for idx in range(line_pairs.shape[0]):
            i = int(line_pairs[idx, 0])
            j = int(line_pairs[idx, 1])
            a1 = line_pairs[idx, 2] 
            a2 = line_pairs[idx, 3] 
            if a1 > math.pi:
                a1 -= math.pi
            e1 = abs(a1 - math.pi / 2)
            if a2 > math.pi:
                a2 -= math.pi
            e2 = abs(a2 - math.pi / 2)
            e = max(e1, e2)
            if e < ANGLE_THRESHOLD:
                line11 = np.reshape(m_lines1_inl[i], (4, 1))
                line12 = np.reshape(m_lines1_inl[j], (4, 1))
                line21 = np.reshape(m_lines2_inl[i], (4, 1))
                line22 = np.reshape(m_lines2_inl[j], (4, 1))

                junctions_1.append(_estimators.Junction2d(line11, line12))
                junctions_2.append(_estimators.Junction2d(line21, line22))

                if len(junctions_1) >= point_matches.shape[0] + MAX_JUNCTIONS:
                    break

    # Add only those lines that are almost orthogonal
    if USE_JUNCTIONS:
        #junc1, junc2, indices = get_actual_junctions(m_lines1_inl, m_lines2_inl, K1, K2, filter_threshold=3)
        #print(junc1.shape)
        if True and junc1 is not None:
            for idx in range(junc1.shape[0]):
                junctions_1.append(_estimators.Junction2d(junc1[idx][:2]))
                junctions_2.append(_estimators.Junction2d(junc2[idx][:2]))
        if False and junc_indices is not None:
            for [idx1, idx2] in junc_indices:
                line11 = np.reshape(m_lines1_inl[idx1], (4, 1))
                line12 = np.reshape(m_lines1_inl[idx2], (4, 1))
                line21 = np.reshape(m_lines2_inl[idx1], (4, 1))
                line22 = np.reshape(m_lines2_inl[idx2], (4, 1))
                junctions_1.append(_estimators.Junction2d(line11, line12))
                junctions_2.append(_estimators.Junction2d(line21, line22))

    if isinstance(m_vp1, list):
        m_label1 = np.zeros((0))
        m_label2 = np.zeros((0))
        m_vp1 = np.zeros((3, 0))
        m_vp2 = np.zeros((3, 0))
    elif np.array(m_vp1).shape[0] > 0:
        m_vp1 = m_vp1.transpose()
        m_vp2 = m_vp2.transpose()
    else:
        m_vp1 = np.zeros((3, 0))
        m_vp2 = np.zeros((3, 0))
        m_label1 = np.zeros((0))
        m_label2 = np.zeros((0))
        
    if m_lines1.shape[0] >= 2:
        m_lines1_inl = m_lines1_inl.reshape(m_lines1_inl.shape[0], -1).transpose()
        m_lines2_inl = m_lines2_inl.reshape(m_lines2_inl.shape[0], -1).transpose()
    else:
        m_lines1_inl = np.zeros((4, 0))
        m_lines2_inl = np.zeros((4, 0))
        
    start_time = time.time()
    pred_R_1_2, pred_T_1_2, pred_E_1_2 = run_hybrid_relative_pose(
        K1, K2,
        [m_lines1_inl, m_lines2_inl],
        [m_vp1, m_vp2],
        [junctions_1, junctions_2],
        [m_label1, m_label2],
        th_vp_angle=TH_VP_ANGLE,
        th_pixel=TH_PIXEL,
        data_weights=DATA_WEIGHTS,
        solver_flags=SOLVER_FLAGS,
        ls_refinement=LS_REFINEMENT,
        weights_refinement=WEIGHTS_REFINEMENT,
        line_inlier_ratio=LINE_INLIER_RATIO)
    elapsed_time = time.time() - start_time
    if CORE_NUMBER < 2:
        print(f"Estimation time = {elapsed_time * 1000:.2f} ms")
    return evaluate_R_t(gt_R_1_2, gt_T_1_2, pred_R_1_2, pred_T_1_2), elapsed_time

processing_queue = []
pose_errors = []
runtimes = []
all_results = {}
run_count = 1
print(f"Collecting data for [{run_count * BATCH_SIZE} / {len(dataloader)}] pairs")
for i, data in enumerate(dataloader):
    if i % 100 == 0:
        print(f"{i} / {min(len(dataloader), BATCH_SIZE)}")
    point_matches, m_lines1, m_lines2, m_junc1, m_junc2, m_junc_indices, m_line_pairs = detect_and_load_data(data, line_matcher, CORE_NUMBER)
    processing_queue.append((data, point_matches, m_lines1, m_lines2, m_junc1, m_junc2, m_junc_indices, m_line_pairs))
    # Running the estimators so we don't have too much things in the memory
    if len(processing_queue) >= BATCH_SIZE or i == len(dataloader) - 1:
        for SOLVER_FLAGS, TH_VP_ANGLE, TH_PIXEL, MAX_JUNCTIONS, ANGLE_THRESHOLD, USE_JOINT_VP_MATCHING, USE_ENDPOINTS, USE_JUNCTIONS, REFINE_VP, LS_REFINEMENT, LINE_INLIER_RATIO in SOLVERSs:
            ANGLE_THRESHOLD = ANGLE_THRESHOLD / 180.0 * math.pi
            print(f"Pose estimation...")
            results = Parallel(n_jobs=min(CORE_NUMBER, len(processing_queue)))(delayed(process_pair)(
                data,
                point_matches,
                m_lines1,
                m_lines2,
                m_junc1,
                m_junc2,
                m_junc_indices,
                m_line_pairs,
                CORE_NUMBER,
                SOLVER_FLAGS,
                TH_PIXEL,
                USE_ENDPOINTS,
                USE_JUNCTIONS,
                REFINE_VP,
                ANGLE_THRESHOLD,
                USE_JOINT_VP_MATCHING,
                TH_VP_ANGLE,
                LS_REFINEMENT,
                LINE_INLIER_RATIO) for data, point_matches, m_lines1, m_lines2, m_junc1, m_junc2, m_junc_indices, m_line_pairs in tqdm(processing_queue))

            key = str([SOLVER_FLAGS, TH_VP_ANGLE, TH_PIXEL, MAX_JUNCTIONS, ANGLE_THRESHOLD, USE_JOINT_VP_MATCHING, USE_ENDPOINTS, USE_JUNCTIONS, REFINE_VP, LS_REFINEMENT, LINE_INLIER_RATIO])
            if key not in all_results.keys():
                all_results[key] = []
            all_results[key].append(results)
            print(key)

        # Clearing the processing queue
        processing_queue = []
        if run_count * BATCH_SIZE < dataset.__len__():
            run_count += 1
            print(f"Collecting data for [{min(dataset.__len__(), run_count * BATCH_SIZE)} / {dataset.__len__()}] pairs")

            
for key in all_results.keys():
    SOLVER_FLAGS, TH_VP_ANGLE, TH_PIXEL, MAX_JUNCTIONS, ANGLE_THRESHOLD, USE_JOINT_VP_MATCHING, USE_ENDPOINTS, USE_JUNCTIONS, REFINE_VP, LS_REFINEMENT, LINE_INLIER_RATIO = ast.literal_eval(key)
                    
    # Concatenating the results to the main lists
    pose_errors = []
    runtimes = []
    for results in all_results[key]:
        pose_errors += [error for error, time in results]
        runtimes += [time for error, time in results]

    pose_errors = np.array(pose_errors)

    print(f"Average run-time: {1000 * np.mean(runtimes):.2f} ms")
    aucR = 100 * np.r_[pose_auc(pose_errors[:,0], thresholds=[2, 5, 10, 20])]
    print(f"Median rotation error: {np.median(pose_errors[:,0]):.2f}")
    print(f"AUC at 2 / 5 / 10 / 20 deg error: {aucR[0]:.2f} / {aucR[1]:.2f} / {aucR[2]:.2f} / {aucR[3]:.2f}")
    auct = 100 * np.r_[pose_auc(pose_errors[:,1], thresholds=[2, 5, 10, 20])]
    print(f"Median translation error: {np.median(pose_errors[:,1]):.2f}")
    print(f"AUC at 2 / 5 / 10 / 20 deg error: {auct[0]:.2f} / {auct[1]:.2f} / {auct[2]:.2f} / {auct[3]:.2f}")
    aucRt = 100 * np.r_[pose_auc(pose_errors.max(1), thresholds=[2, 5, 10, 20])]
    print(f"Median pose error: {np.median(pose_errors.max(1)):.2f}")
    print(f"AUC at 2 / 5 / 10 / 20 deg error: {aucRt[0]:.2f} / {aucRt[1]:.2f} / {aucRt[2]:.2f} / {aucRt[3]:.2f}")

    with open('phototourism_results_after_iccv.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([MATCHER, matcher_type, line_method] + SOLVER_FLAGS + [LINE_CUT_LENGTH, TH_VP_ANGLE, TH_PIXEL, MAX_JUNCTIONS, ANGLE_THRESHOLD, USE_JOINT_VP_MATCHING, USE_ENDPOINTS, USE_JUNCTIONS, REFINE_VP, LS_REFINEMENT, LINE_INLIER_RATIO, round(1000 * np.mean(runtimes), 2), round(np.median(pose_errors), 2)] + list(np.round(aucRt, 2)))
        