#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_2VP_3CLL_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_2VP_3CLL_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_2vp_2pt.h"

namespace line_relative_pose {

// 2 VPs and 3 lines
// [assumption] Two pairs among the 3 lines are coplanar
class RelativePoseSolver2vp3cll: public RelativePoseSolver2vp2pt {
public:
    RelativePoseSolver2vp3cll(): RelativePoseSolver2vp2pt() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    std::vector<int> min_sample_size() const override { return {3, 2, 0}; }

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

