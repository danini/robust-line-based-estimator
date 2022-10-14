#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_3CLL_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_3CLL_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_1vp_3pt.h"

namespace line_relative_pose {

// 1 VP + 3 coplar lines 
class RelativePoseSolver1vp3cll: public RelativePoseSolver1vp3pt {
public:
    RelativePoseSolver1vp3cll(): RelativePoseSolver1vp3pt() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    std::vector<int> min_sample_size() const override { return {3, 1, 0}; }

    int MinimalSolverWrapper(const std::vector<LineMatch>& line_matches, 
                             const std::vector<VPMatch>& vp_matches,
                             const std::vector<JunctionMatch>& junction_matches,
                             std::vector<ResultType>* res) const override;

protected:
    int MinimalSolver(const std::vector<LineMatch>& line_matches,
                      const std::vector<VPMatch>& vp_matches,
                      std::vector<ResultType>* res) const;
};

}  // namespace line_relative_pose 

#endif 

