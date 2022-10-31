import numpy as np
import sys
sys.path.append("build/robust_line_based_estimator")
import line_relative_pose_estimators as _estimators

def run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, th_vp_angle=5.0, th_pixel=1.0, data_weights=[0.0, 0.0, 1.0], solver_flags=[True, True, True, True, True, True, True, True, True], ls_refinement=0, weights_refinement=[1.0, 1.0], line_inlier_ratio=0.3):
    '''
    Inputs:
    - line_matches: pair of numpy array the same size [(4, n_lines), (4, n_lines)]
    - vp_matches: pair of numpy array with the same size [(3, n_vps), (3, n_vps)]
    - junction_matches: pair of list [vector<_estimators.Junction2d>, vector<_estimators.Junction2d>]
    - vp_labels: pair of list [vector<int>, vector<int>]
    - th_pixel: threshold
    - solver_flags: whether to enable each solver
    '''
    if len(vp_matches[0].shape) != 2 or vp_matches[0].shape[1] < 2:
        solver_flags[3] = False
    if len(vp_matches[0].shape) != 2 or vp_matches[0].shape[1] < 1:
        solver_flags[2] = False

    threshold_normalizer = 0.25 * (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1])
    th_pixel = th_pixel / threshold_normalizer

    if any(solver_flags[1:]) == True:
        data_weights[1] = 1.0 / threshold_normalizer
    else:
        th_vp_angle = 0
        data_weights[1] = 0
        line_matches = [np.zeros((4, 0)), np.zeros((4, 0))]
        vp_matches = [np.zeros((3, 0)), np.zeros((3, 0))]
        vp_labels = [np.zeros((0, 1)), np.zeros((0, 1))]

    if any(solver_flags) == True:
        options = _estimators.HybridLORansacOptions()
        options.data_type_weights_ = np.array(data_weights)
        options.squared_inlier_thresholds_ = np.array([1.0, th_vp_angle, th_pixel])
        # options.final_least_squares_ = True
        res = _estimators.run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, options, solver_flags, ls_refinement, weights_refinement, line_inlier_ratio)
        return res[0][0], res[0][1], res[0][2]
    else:
        return np.eye(3), np.zeros((3)), np.eye(3)

