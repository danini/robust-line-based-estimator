#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_2VP_2PT_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_2VP_2PT_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_base.h"

namespace line_relative_pose {

// 2 VPs + 2 points
class RelativePoseSolver2vp2pt: public RelativePoseSolverBase {
public:
    RelativePoseSolver2vp2pt(): RelativePoseSolverBase() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    std::vector<int> min_sample_size() const override { return {0, 2, 2}; }

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

