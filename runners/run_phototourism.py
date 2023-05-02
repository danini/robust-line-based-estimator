#!/usr/bin/env python
# coding: utf-8
# ## Relative pose estimation on ScanNet
import os, sys
import cv2
import numpy as np
from tqdm import tqdm
import time
import math
from joblib import Parallel, delayed

from robust_line_based_estimator.datasets.phototourism import PhotoTourism
from robust_line_based_estimator.line_matching.line_matcher import LineMatcher
from robust_line_based_estimator.line_matching.gluestick import GlueStick
from kornia.feature import LoFTR
from robust_line_based_estimator.vp_matcher import vp_matching, associate_lines_to_vps
from robust_line_based_estimator.evaluation import evaluate_R_t, pose_auc
from third_party.SuperGluePretrainedNetwork.models.matching import Matching
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import verify_pyprogressivex, point_matching, joint_vp_detection_and_matching
from robust_line_based_estimator.hybrid_relative_pose import run_hybrid_relative_pose
from robust_line_based_estimator.line_junction_utils import append_h5, read_h5, get_endpoint_correspondences, angular_check
from robust_line_based_estimator.point_based_relative_pose import run_point_based_relative_pose

sys.path.append("build/robust_line_based_estimator")
import line_relative_pose_estimators as _estimators

###########################################
# Hyperparameters to be tuned
###########################################
TH_PIXEL = 3.0
ANGLE_THRESHOLD = math.pi / 32
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
REFINE_VP_WITH_ALL_LINES = True
MATCHER = "GS"  # "SG", "LoFTR", or "GS"
OUTPUT_DB_PATH = "phototourism_matches.h5"
CORE_NUMBER = 16
BATCH_SIZE = 100

###########################################
# Initialize the dataset
###########################################
dataset = PhotoTourism(
    root_dir=os.path.expanduser("/home/remi/Documents/datasets/RANSAC-Tutorial-Data"),
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
    matcher = GlueStick({'device': device})
    matcher_key = 'gs'
else:
    raise ValueError("Unknown matcher " + MATCHER)

###########################################
# Initialize the line method
###########################################
line_method = 'deeplsd'  # 'lsd', 'SOLD2', or 'deeplsd' supported for now
matcher_type  = 'superglue_endpoints'  # 'lbd', 'sold2', 'superglue_endpoints', or 'gluestick'
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
            'weights': 'outdoor'
        }
    }
    line_matcher = LineMatcher(line_detector=line_method, line_matcher='superglue_endpoints', conf=conf)
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
    lines_1 = read_h5(f"{label}-1", OUTPUT_DB_PATH)
    lines_2 = read_h5(f"{label}-2", OUTPUT_DB_PATH)
    m_lines1 = read_h5(f"{label}-m1", OUTPUT_DB_PATH)
    m_lines2 = read_h5(f"{label}-m2", OUTPUT_DB_PATH)

    if (lines_1 is None or lines_2 is None
        or m_lines1 is None or m_lines2 is None):
        # Detect lines in the images
        line_feat1 = line_matcher.detect_and_describe_lines(gray_img1)
        line_feat2 = line_matcher.detect_and_describe_lines(gray_img2)
        elapsed_time = time.time() - start_time
        if CORE_NUMBER < 2:
            print(f"Line detection detection time = {elapsed_time * 1000:.2f} ms")

        # Match lines in the images
        start_time = time.time()
        _, m_lines1, m_lines2 = line_matcher.match_lines(
            gray_img1, gray_img2, line_feat1, line_feat2)
        elapsed_time = time.time() - start_time
        if CORE_NUMBER < 2:
            print(f"Line matching detection time = {elapsed_time * 1000:.2f} ms")

        # Saving to the database
        lines_1 = line_feat1["line_segments"]
        lines_2 = line_feat2["line_segments"]
        append_h5({f"{label}-1": lines_1, f"{label}-2": lines_2,
                   f"{label}-m1": m_lines1, f"{label}-m2": m_lines2},
                  OUTPUT_DB_PATH)
    else:
        elapsed_time = time.time() - start_time
        if CORE_NUMBER < 2:
            print(f"Line detection/matching detection time = {elapsed_time * 1000:.2f} ms")

    return point_matches, lines_1, lines_2, m_lines1, m_lines2

###########################################
# Relative pose estimation
###########################################
def process_pair(data, point_matches, lines_1, lines_2,
                 m_lines1, m_lines2, CORE_NUMBER):
    gt_R_1_2 = data["R_1_2"]
    gt_T_1_2 = data["T_1_2"]
    K1 = data["K1"]
    K2 = data["K2"]

    # Run point-based estimation if no line correspondences are found
    if m_lines1.shape[0] < 2:
        start_time = time.time()
        R, t = run_point_based_relative_pose(K1, K2,
            point_matches,
            np.array([]),
            th_pixel = TH_PIXEL,
            config = 0)
        elapsed_time = time.time() - start_time
        return evaluate_R_t(gt_R_1_2, gt_T_1_2, R, t), elapsed_time

    # Compute and match VPs or load them from the database
    m_lines1_inl = m_lines1[:, :, [1, 0]]
    m_lines2_inl = m_lines2[:, :, [1, 0]]

    start_time = time.time()
    if USE_JOINT_VP_MATCHING:
        # Jointly detect VP and match them
        m_vp1, m_vp2, m_label1 = joint_vp_detection_and_matching(
            int(2 * K1[0,2]), int(2 * K1[1,2]), m_lines1_inl,
            m_lines2_inl, threshold=1.5)
        m_label2 = m_label1
    else:
        # Detect vanishing points in the source image
        vp1, vp_label1 = verify_pyprogressivex(
            int(2 * K1[0,2]), int(2 * K1[1,2]),
            m_lines1_inl, threshold=1.5)
        # Detect vanishing points in the destination image
        vp2, vp_label2 = verify_pyprogressivex(
            int(2 * K2[0,2]), int(2 * K2[1,2]),
            m_lines2_inl, threshold=1.5)
        # Matching the vanishing points
        m_vp1, m_label1, m_vp2, m_label2 = vp_matching(vp1, vp_label1,
                                                       vp2, vp_label2)

    if REFINE_VP:
        if len(m_vp1) > 0:
            if REFINE_VP_WITH_ALL_LINES:
                # Compute the VP labels with all lines
                label_1 = associate_lines_to_vps(lines_1, m_vp1)
                # Refine the VPs
                m_vp1 = _estimators.refine_vp(
                    label_1, lines_1[:, :, [1, 0]].reshape(-1, 4, 1), m_vp1)
            else:
                m_vp1 = _estimators.refine_vp(
                    m_label1, m_lines1_inl.reshape(-1, 4, 1), m_vp1)
        m_vp1 = np.array(m_vp1)
        if len(m_vp2) > 0:
            if REFINE_VP_WITH_ALL_LINES:
                label_2 = associate_lines_to_vps(lines_2, m_vp2)
                m_vp2 = _estimators.refine_vp(
                    label_2, lines_2[:, :, [1, 0]].reshape(-1, 4, 1), m_vp2)
            else:
                m_vp2 = _estimators.refine_vp(
                    m_label2, m_lines2_inl.reshape(-1, 4, 1), m_vp2)
        m_vp2 = np.array(m_vp2)
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
        return evaluate_R_t(gt_R_1_2, gt_T_1_2, R, t), elapsed_time

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
    line_pairs = angular_check(m_lines1, m_lines2, K1, K2)
    for i, j, a1, a2 in line_pairs:
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

    start_time = time.time()
    pred_R_1_2, pred_T_1_2, pred_E_1_2 = run_hybrid_relative_pose(
        K1, K2,
        [m_lines1_inl.reshape(m_lines1_inl.shape[0], -1).transpose(),
         m_lines2_inl.reshape(m_lines2_inl.shape[0], -1).transpose()],
        [m_vp1.transpose(), m_vp2.transpose()],
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
run_count = 1
for i, data in enumerate(dataloader):
    point_matches, lines_1, lines_2, m_lines1, m_lines2 = detect_and_load_data(
        data, line_matcher, CORE_NUMBER)
    processing_queue.append((data, point_matches, lines_1, lines_2,
                             m_lines1, m_lines2))
    # Running the estimators so we don't have too much things in the memory
    if len(processing_queue) >= BATCH_SIZE or i == len(dataloader) - 1:
        print(f"Pose estimation...")
        results = Parallel(n_jobs=min(CORE_NUMBER, len(processing_queue)))(delayed(process_pair)(
            data,
            point_matches,
            lines_1,
            lines_2,
            m_lines1,
            m_lines2,
            CORE_NUMBER) for data, point_matches, lines_1, lines_2, m_lines1, m_lines2 in tqdm(processing_queue))

        # Concatenating the results to the main lists
        pose_errors += [error for error, time in results]
        runtimes += [time for error, time in results]

        # Clearing the processing queue
        processing_queue = []
        run_count += 1
        print(f"Collecting data for [{run_count * BATCH_SIZE} / {len(dataloader)}] pairs")
pose_errors = np.array(pose_errors)

print(f"Average run-time: {1000 * np.mean(runtimes):.2f} ms")
auc = 100 * np.r_[pose_auc(pose_errors[:,0], thresholds=[5, 10, 20])]
print(f"Median rotation error: {np.median(pose_errors[:,0]):.2f}")
print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")
auc = 100 * np.r_[pose_auc(pose_errors[:,1], thresholds=[5, 10, 20])]
print(f"Median translation error: {np.median(pose_errors[:,1]):.2f}")
print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")
auc = 100 * np.r_[pose_auc(pose_errors.max(1), thresholds=[5, 10, 20])]
print(f"Median pose error: {np.median(pose_errors.max(1)):.2f}")
print(f"AUC at 5 / 10 / 20 deg error: {auc[0]:.2f} / {auc[1]:.2f} / {auc[2]:.2f}")
