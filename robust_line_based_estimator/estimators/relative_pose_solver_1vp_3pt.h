#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_3PT_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_3PT_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_base.h"

namespace line_relative_pose {

class RelativePoseSolver1vp3pt: public RelativePoseSolverBase {
public:
    RelativePoseSolver1vp3pt(): RelativePoseSolverBase() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    std::vector<int> min_sample_size() const override { return {0, 1, 3}; }

    int MinimalSolverWrapper(const std::vector<LineMatch>& line_matches, 
                             const std::vector<VPMatch>& vp_matches,
                             const std::vector<JunctionMatch>& junction_matches,
                             std::vector<ResultType>* res) const override;

protected:
    int MinimalSolver(const std::vector<VPMatch>& vp_matches,
                      const std::vector<JunctionMatch>& junction_matches,
                      std::vector<ResultType>* res) const;
};

}  // namespace line_relative_pose 

#endif 

