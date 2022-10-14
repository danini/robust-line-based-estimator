#include "estimators/relative_pose_solver_2vp_3cll.h"
#include "solvers/solver_2vp_2pt.h"

namespace line_relative_pose {

int RelativePoseSolver2vp3cll::MinimalSolver(const std::vector<LineMatch>& line_matches,
                                            const std::vector<VPMatch>& vp_matches,
                                            std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 3);
    THROW_CHECK_EQ(vp_matches.size(), 2);

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
    
    // solve for 2vp + 2pt
    int num_sols = 0;
    int tmp_num_sols;
    std::vector<ResultType> tmpres;
    std::vector<JunctionMatch> jmatches;

    // junc0 - junc1
    tmpres.clear();
    jmatches.clear();
    jmatches.push_back(junction_matches[0]);
    jmatches.push_back(junction_matches[1]);
    tmp_num_sols = RelativePoseSolver2vp2pt::MinimalSolver(vp_matches, jmatches, &tmpres);
    res->insert(res->end(), tmpres.begin(), tmpres.end());
    num_sols += tmp_num_sols;

    // junc0 - junc2
    tmpres.clear();
    jmatches.clear();
    jmatches.push_back(junction_matches[0]);
    jmatches.push_back(junction_matches[2]);
    tmp_num_sols = RelativePoseSolver2vp2pt::MinimalSolver(vp_matches, jmatches, &tmpres);
    res->insert(res->end(), tmpres.begin(), tmpres.end());
    num_sols += tmp_num_sols;

    // junc1 - junc2
    tmpres.clear();
    jmatches.clear();
    jmatches.push_back(junction_matches[1]);
    jmatches.push_back(junction_matches[2]);
    tmp_num_sols = RelativePoseSolver2vp2pt::MinimalSolver(vp_matches, jmatches, &tmpres);
    res->insert(res->end(), tmpres.begin(), tmpres.end());
    num_sols += tmp_num_sols;
    return num_sols;
}

int RelativePoseSolver2vp3cll::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 3);
    THROW_CHECK_EQ(vp_matches.size(), 2);
    THROW_CHECK_EQ(junction_matches.size(), 0);

    return MinimalSolver(line_matches, vp_matches, res);
}

}  // namespace line_relative_pose 

