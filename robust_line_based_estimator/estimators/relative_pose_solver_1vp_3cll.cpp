#include "estimators/relative_pose_solver_1vp_3cll.h"
#include "solvers/solver_1vp_3pt.h"

namespace line_relative_pose {

int RelativePoseSolver1vp3cll::MinimalSolver(const std::vector<LineMatch>& line_matches,
                                            const std::vector<VPMatch>& vp_matches,
                                            std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 3);
    THROW_CHECK_EQ(vp_matches.size(), 1);

    // generate junctions
    std::vector<JunctionMatch> junction_matches;
    Junction2d junc1, junc2;
    // 0 - 1
    junc1 = Junction2d(line_matches[0].first, line_matches[1].first);
    junc2 = Junction2d(line_matches[0].second, line_matches[1].second);
    junction_matches.push_back(std::make_pair(junc1, junc2));
    // 0 - 2
    junc1 = Junction2d(line_matches[0].first, line_matches[2].first);
    junc2 = Junction2d(line_matches[0].second, line_matches[2].second);
    junction_matches.push_back(std::make_pair(junc1, junc2));
    // 1 - 2
    junc1 = Junction2d(line_matches[1].first, line_matches[2].first);
    junc2 = Junction2d(line_matches[1].second, line_matches[2].second);
    junction_matches.push_back(std::make_pair(junc1, junc2));
    
    // solve for 1vp + 3pt
    return RelativePoseSolver1vp3pt::MinimalSolver(vp_matches, junction_matches, res);
}

int RelativePoseSolver1vp3cll::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 3);
    THROW_CHECK_EQ(vp_matches.size(), 1);
    THROW_CHECK_EQ(junction_matches.size(), 0);

    return MinimalSolver(line_matches, vp_matches, res);
}

}  // namespace line_relative_pose 

