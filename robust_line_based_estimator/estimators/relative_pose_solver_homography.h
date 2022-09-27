#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_HOMOGRAPHY_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_HOMOGRAPHY_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_base.h"

namespace line_relative_pose {

// homography-based essential matrix
class RelativePoseSolverHomography: public RelativePoseSolverBase {
public:
    RelativePoseSolverHomography(): RelativePoseSolverBase() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    int MinimalSolverWrapper(const std::vector<LineMatch>& line_matches, 
                             const std::vector<VPMatch>& vp_matches,
                             const std::vector<PointMatch>& junction_matches,
                             std::vector<ResultType>* res) const override;

    virtual int HomographySolver(const std::vector<LineMatch>& line_matches, 
                                 const std::vector<VPMatch>& vp_matches,
                                 const std::vector<PointMatch>& junction_matches,
                                 std::vector<M3D>* Hs) const = 0;

protected:
    int DecomposeHomography(const std::vector<M3D>& Hs, std::vector<ResultType>* res) const;
};

}  // namespace line_relative_pose 

#endif 

