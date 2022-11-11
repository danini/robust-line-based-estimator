#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_2LINE_1PT_ORTHOGONAL_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1VP_2LINE_1PT_ORTHOGONAL_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_1line_1vp_2pt_orthogonal.h"

namespace line_relative_pose {

// 1 vp + 2 points (the first point needs to be an intersection)
// [assumption] the VP is orthogonal to the first line for the first intersection
class RelativePoseSolver1vp2line1pt_orthogonal: public RelativePoseSolver1line1vp2pt_orthogonal {
public:
    RelativePoseSolver1vp2line1pt_orthogonal(): RelativePoseSolver1line1vp2pt_orthogonal() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    std::vector<int> min_sample_size() const override { return {2, 1, 1}; }

    int MinimalSolverWrapper(const std::vector<LineMatch>& line_matches, 
                             const std::vector<VPMatch>& vp_matches,
                             const std::vector<JunctionMatch>& junction_matches,
                             std::vector<ResultType>* res) const override;

protected:
    int MinimalSolver(const std::vector<LineMatch>& line_matches,
                      const std::vector<VPMatch>& vp_matches,
                      const std::vector<JunctionMatch>& junction_matches,
                      std::vector<ResultType>* res) const;
};

}  // namespace line_relative_pose 

#endif 

