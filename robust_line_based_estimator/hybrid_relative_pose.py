import numpy as np
import sys
import copy

sys.path.append("build/robust_line_based_estimator")
import line_relative_pose_estimators as _estimators

solver_scores = {}

def run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, th_vp_angle=5.0, th_pixel=1.0, data_weights=[0.0, 0.0, 1.0], solver_flags=[True, True, True, True, True, True, True, True], ls_refinement=0, weights_refinement=[1.0, 1.0], line_inlier_ratio=0.3):
    '''
    Inputs:
    - line_matches: pair of numpy array the same size [(4, n_lines), (4, n_lines)]
    - vp_matches: pair of numpy array with the same size [(3, n_vps), (3, n_vps)]
    - junction_matches: pair of list [vector<_estimators.Junction2d>, vector<_estimators.Junction2d>]
    - vp_labels: pair of list [vector<int>, vector<int>]
    - th_pixel: threshold
    - solver_flags: whether to enable each solver
    '''
    threshold_normalizer = 0.25 * (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1])
    th_pixel = th_pixel / threshold_normalizer
        
    solver_flags_cpy = copy.deepcopy(solver_flags)
        
    if line_matches[0].shape[1] < 4:
        solver_flags_cpy[2] = False
        solver_flags_cpy[3] = False
        solver_flags_cpy[4] = False
        solver_flags_cpy[6] = False
        solver_flags_cpy[7] = False

    # Turn off the 2VP2PT estimator if there are not enough VPs
    if vp_matches[0].shape[1] < 2:
        solver_flags_cpy[5] = False
    if vp_matches[0].shape[1] < 1:
        solver_flags_cpy[3] = False
        solver_flags_cpy[4] = False
        solver_flags_cpy[6] = False
        solver_flags_cpy[7] = False

    if any(solver_flags[1:]) == True:
        data_weights[1] = 1.0 / threshold_normalizer
    else:
        th_vp_angle = 0
        data_weights[1] = 0
        line_matches = [np.zeros((4, 0)), np.zeros((4, 0))]
        vp_matches = [np.zeros((3, 0)), np.zeros((3, 0))]
        vp_labels = [np.zeros((0, 1)), np.zeros((0, 1))]

    if any(solver_flags_cpy) == True:
        options = _estimators.HybridLORansacOptions()
        options.data_type_weights_ = np.array(data_weights)
        options.squared_inlier_thresholds_ = np.array([1.0, th_vp_angle, th_pixel])
        options.min_num_iterations_ = 20
        options.max_num_iterations_ = 1000
        options.final_least_squares_ = False
        options.num_lo_steps_ = 0
        options.num_lsq_iterations_ = 0
        res = _estimators.run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, options, solver_flags_cpy, ls_refinement, weights_refinement, 0.5)
        
        #.def_readwrite("num_iterations_total", &ransac_lib::HybridRansacStatistics::num_iterations_total)
        #.def_readwrite("num_iterations_per_solver", &ransac_lib::HybridRansacStatistics::num_iterations_per_solver)
        #.def_readwrite("best_num_inliers", &ransac_lib::HybridRansacStatistics::best_num_inliers)
        #.def_readwrite("best_solver_type", &ransac_lib::HybridRansacStatistics::best_solver_type)
        #.def_readwrite("best_model_score", &ransac_lib::HybridRansacStatistics::best_model_score)
        #.def_readwrite("inlier_ratios", &ransac_lib::HybridRansacStatistics::inlier_ratios)
        #.def_readwrite("inlier_indices", &ransac_lib::HybridRansacStatistics::inlier_indices)
        #.def_readwrite("number_lo_iterations", &ransac_lib::HybridRansacStatistics::number_lo_iterations);
        return res[0][0], res[0][1], res[0][2]
    else:
        return np.eye(3), np.zeros((3)), np.eye(3)

