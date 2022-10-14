#include "estimators/relative_pose_solver_1vp_2pt_orthogonal.h"

namespace line_relative_pose {

int RelativePoseSolver1vp2pt_orthogonal::MinimalSolver(const std::vector<VPMatch>& vp_matches,
                                                       const std::vector<JunctionMatch>& junction_matches,
                                                       std::vector<ResultType>* res) const 
{
    // The first junction needs to be an intersection
    if (!junction_matches[0].first.IsJunction() || !junction_matches[0].second.IsJunction())
        return 0;

    std::vector<LineMatch> line_matches;
    line_matches.push_back(std::make_pair(junction_matches[0].first.line1(), junction_matches[0].second.line1()));
    return RelativePoseSolver1line1vp2pt_orthogonal::MinimalSolver(line_matches, vp_matches, junction_matches, res);
}

int RelativePoseSolver1vp2pt_orthogonal::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 0);
    THROW_CHECK_EQ(vp_matches.size(), 1);
    THROW_CHECK_EQ(junction_matches.size(), 2);

    return MinimalSolver(vp_matches, junction_matches, res);
}

}  // namespace line_relative_pose 

