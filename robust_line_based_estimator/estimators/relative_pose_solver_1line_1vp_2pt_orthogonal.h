#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1LINE_1VP_2PT_ORTHOGONAL_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_1LINE_1VP_2PT_ORTHOGONAL_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_2vp_2pt.h"

namespace line_relative_pose {

class RelativePoseSolver1line1vp2pt_orthogonal: public RelativePoseSolver2vp2pt {
public:
    RelativePoseSolver1line1vp2pt_orthogonal(): RelativePoseSolver2vp2pt() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    std::vector<int> min_sample_size() const override { return {1, 1, 2}; }

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

