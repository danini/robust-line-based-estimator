import numpy as np
import sys
sys.path.append("build/robust_line_based_estimator")
import robust_line_based_estimator.estimators as _estimators

def run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, th_pixel=1.0, solver_flags=[True, True, True, True]):
    options = _estimators.HybridLORansacOptions()
    options.data_type_weights_ = np.array([0.0, 0.0, 1.0])
    options.squared_inlier_thresholds_ = np.array([0.0, 0.0, th_pixel])
    res = _estimators.run_hybrid_relative_pose(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, options, solver_flags)
    return res[0][0], res[0][1]

