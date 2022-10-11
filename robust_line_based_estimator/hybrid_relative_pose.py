import numpy as np
import sys
sys.path.append("build/robust_line_based_estimator")
import line_relative_pose_estimators as _estimators

def run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, th_pixel=1.0, solver_flags=[True, True, True, True]):
    '''
    Inputs:
    - line_matches: pair of numpy array the same size [(4, n_lines), (4, n_lines)]
    - vp_matches: pair of numpy array with the same size [(3, n_vps), (3, n_vps)]
    - junction_matches: pair of list [vector<_base.Junction2d>, vector<_base.Junction2d>]
    - vp_labels: pair of list [vector<int>, vector<int>]
    - th_pixel: threshold
    - solver_flags: whether to enable each solver
    '''
    options = _estimators.HybridLORansacOptions()
    options.data_type_weights_ = np.array([0.0, 0.0, 1.0])
    options.squared_inlier_thresholds_ = np.array([0.0, 0.0, th_pixel])
    res = _estimators.run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, options, solver_flags)
    return res[0][0], res[0][1]

