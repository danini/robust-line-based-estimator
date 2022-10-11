#ifndef BASE_TYPES_H_
#define BASE_TYPES_H_

#include <Eigen/Core>
#include <limits>
#include "base/linebase.h"
#include "base/log_exceptions.h"

namespace line_relative_pose {

const double EPS = std::numeric_limits<double>::min();
using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V4D = Eigen::Vector4d;
using M2D = Eigen::Matrix2d;
using M3D = Eigen::Matrix3d;

inline V3D homogeneous(const V2D& v2d) { return V3D(v2d(0), v2d(1), 1.0); }
inline V2D dehomogeneous(const V3D& v3d) { return V2D(v3d(0) / (v3d(2) + EPS), v3d(1) / (v3d(2) + EPS)); }

typedef std::pair<limap::Line2d, limap::Line2d> LineMatch;
typedef std::pair<V3D, V3D> VPMatch;

} // namespace line_relative_pose 

#endif

