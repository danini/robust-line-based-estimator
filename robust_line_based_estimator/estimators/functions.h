#ifndef LINE_RELATIVE_POSE_ESTIMATORS_FUNCTIONS_H_
#define LINE_RELATIVE_POSE_ESTIMATORS_FUNCTIONS_H_

#include <iostream>
#include <utility>
#include <Eigen/Core>
#include <RansacLib/hybrid_ransac.h>

#include "base/types.h"
#include "base/junction.h"

namespace line_relative_pose {

// for each of the two input lines in line_matches: 4 x n array, with each column [x1, y1, x2, y2]
std::pair<std::tuple<M3D, V3D, M3D>, ransac_lib::HybridRansacStatistics> run_hybrid_relative_pose(
        const M3D& K1, const M3D& K2,
        const std::pair<Eigen::Matrix4Xd, Eigen::Matrix4Xd>& line_matches, 
        const std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>& vp_matches, 
        const std::pair<std::vector<Junction2d>, std::vector<Junction2d>>& junction_matches, 
        const std::pair<std::vector<int>, std::vector<int>> vp_labels,
        const ransac_lib::HybridLORansacOptions& options,
        const std::vector<bool>& solver_flags);

} // namespace line_relative_pose 

#endif

