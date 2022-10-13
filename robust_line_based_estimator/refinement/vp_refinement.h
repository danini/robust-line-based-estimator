#ifndef LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_VP_REFINEMENT_H_
#define LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_VP_REFINEMENT_H_

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "base/types.h"
#include "base/linebase.h"

namespace line_relative_pose {

std::vector<V3D> VPRefinement(
    const std::vector<int>& labels,
    const std::vector<V4D>& inlier_lines,
    std::vector<V3D>& vps
);

} // namespace line_relative_pose 

#endif
