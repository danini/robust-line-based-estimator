#ifndef LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_LS_COMBINED_H_
#define LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_LS_COMBINED_H_

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include "base/types.h"
#include "base/junction.h"

namespace line_relative_pose {

void LeastSquares_Combined(const std::vector<JunctionMatch>& junction_matches, 
                           const std::vector<VPMatch>& vp_matches,
                           const std::vector<std::vector<limap::Line2d>>& sup_lines_img1,
                           const std::vector<std::vector<limap::Line2d>>& sup_lines_img2,
                           std::tuple<M3D, V3D, M3D>* res,
                           double weight_vp_rotation,
                           double weight_vp_line,
                           bool fix_vp = false);

} // namespace line_relative_pose 

#endif
