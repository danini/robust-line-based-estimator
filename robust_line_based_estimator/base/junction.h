#ifndef LINE_RELATIVE_POSE_BASE_JUNCTIONS_H
#define LINE_RELATIVE_POSE_BASE_JUNCTIONS_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/linebase.h"
#include "base/types.h"

namespace line_relative_pose  {

class Junction2d {
public:
    Junction2d() {}
    Junction2d(const limap::Line2d& line1, const limap::Line2d& line2) {
        l1 = line1; l2 = line2;
        intersection = l1.coords().cross(l2.coords()).normalized();
        is_junction = true;
    }
    Junction2d(const V4D& line1, const V4D& line2) {
        l1 = limap::Line2d(line1); l2 = limap::Line2d(line2);
        intersection = l1.coords().cross(l2.coords()).normalized();
        is_junction = true;
    }
    Junction2d(const V2D& point) {
        intersection = homogeneous(point);
        is_junction = false;
    }
    
    // utilities
    bool IsJunction() const { return is_junction; }
    V2D point() const { return dehomogeneous(intersection); }
    limap::Line2d line1() const { return l1; }
    limap::Line2d line2() const { return l2; }

private:
    limap::Line2d l1, l2;
    V3D intersection; // homoegeneous coordinate of the intersection
    bool is_junction = false;
};

typedef std::pair<Junction2d, Junction2d> JunctionMatch;

} // namespace line_relative_pose 

#endif

