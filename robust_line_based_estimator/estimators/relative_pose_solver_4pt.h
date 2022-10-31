#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_4PT_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_4PT_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_homography.h"

namespace line_relative_pose {

// 4-point homography solver
// [assumption] 4 coplanar points 
class RelativePoseSolver4pt: public RelativePoseSolverHomography {
public:
    RelativePoseSolver4pt(): RelativePoseSolverHomography() {}
    using ResultType = RelativePoseSolverHomography::ResultType;

    std::vector<int> min_sample_size() const override { return {0, 0, 4}; }

    int HomographySolver(const std::vector<LineMatch>& line_matches, 
                         const std::vector<VPMatch>& vp_matches,
                         const std::vector<JunctionMatch>& junction_matches,
                         std::vector<M3D>* Hs) const override;
};

}  // namespace line_relative_pose 

#endif 

