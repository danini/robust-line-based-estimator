#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_5PT_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_5PT_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_base.h"

namespace line_relative_pose {

// 5-point essential matrix
class RelativePoseSolver5pt: public RelativePoseSolverBase {
public:
    RelativePoseSolver5pt(): RelativePoseSolverBase() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    std::vector<int> min_sample_size() const override { return {0, 0, 5}; }

    int MinimalSolverWrapper(const std::vector<LineMatch>& line_matches, 
                             const std::vector<VPMatch>& vp_matches,
                             const std::vector<PointMatch>& junction_matches,
                             std::vector<ResultType>* res) const override;

protected:
    int MinimalSolver(const std::vector<PointMatch>& point_matches,
                      std::vector<ResultType>* res) const;
};

}  // namespace line_relative_pose 

#endif 

