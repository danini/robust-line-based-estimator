#ifndef LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_LS_SAMPSON_H_
#define LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_LS_SAMPSON_H_

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

void LeastSquares_Sampson(
    const std::vector<JunctionMatch>& junction_matches, std::tuple<M3D, V3D, M3D>* res);

} // namespace line_relative_pose 

#endif
