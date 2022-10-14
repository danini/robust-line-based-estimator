#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_3CLL_ORTHOGONAL_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_3CLL_ORTHOGONAL_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_1line_1vp_2pt_orthogonal.h"

namespace line_relative_pose {

class RelativePoseSolver1vp3cll_orthogonal: public RelativePoseSolver1line1vp2pt_orthogonal {
public:
    RelativePoseSolver1vp3cll_orthogonal(): RelativePoseSolver1line1vp2pt_orthogonal() {}
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

