#include "estimators/relative_pose_solver_1vp_2line_1pt_orthogonal.h"

namespace line_relative_pose {

int RelativePoseSolver1vp2line1pt_orthogonal::MinimalSolver(const std::vector<LineMatch>& line_matches,
                                                            const std::vector<VPMatch>& vp_matches,
                                                            const std::vector<JunctionMatch>& junction_matches,
                                                            std::vector<ResultType>* res) const 
{
    Junction2d j1 = Junction2d(line_matches[0].first, line_matches[1].first);
    Junction2d j2 = Junction2d(line_matches[0].second, line_matches[1].second);

    std::vector<LineMatch> line_matches_new;
    line_matches_new.push_back(line_matches[0]); // we only use the first line match

    std::vector<JunctionMatch> junction_matches_new;
    junction_matches_new.push_back(junction_matches[0]);
    junction_matches_new.push_back(std::make_pair(j1, j2));
    return RelativePoseSolver1line1vp2pt_orthogonal::MinimalSolver(line_matches_new, vp_matches, junction_matches_new, res);
}

int RelativePoseSolver1vp2line1pt_orthogonal::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                                   const std::vector<VPMatch>& vp_matches,
                                                                   const std::vector<JunctionMatch>& junction_matches,
                                                                   std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 2);
    THROW_CHECK_EQ(vp_matches.size(), 1);
    THROW_CHECK_EQ(junction_matches.size(), 1);

    return MinimalSolver(line_matches, vp_matches, junction_matches, res);
}

}  // namespace line_relative_pose 

