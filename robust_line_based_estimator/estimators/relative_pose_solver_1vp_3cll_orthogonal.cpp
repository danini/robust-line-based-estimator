#include "estimators/relative_pose_solver_1vp_3cll_orthogonal.h"
#include "solvers/solver_1vp_3pt.h"

namespace line_relative_pose {

int RelativePoseSolver1vp3cll_orthogonal::MinimalSolver(const std::vector<LineMatch>& line_matches,
                                            const std::vector<VPMatch>& vp_matches,
                                            std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 3);
    THROW_CHECK_EQ(vp_matches.size(), 1);
    res->clear();
    std::vector<VPMatch> new_vp_matches;
    new_vp_matches.push_back(vp_matches[0]);

    // generate the other vp match from orthogonality on the first line
    V3D vq1 = line_matches[0].first.coords().cross(vp_matches[0].first).normalized();
    V3D vq2 = line_matches[0].second.coords().cross(vp_matches[0].second).normalized();
    new_vp_matches.push_back(std::make_pair(vq1, vq2));

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
    
    // solve for 1line + 1vp + 2pt
    std::vector<LineMatch> new_line_matches;
    new_line_matches.push_back(line_matches[0]);
    int num_sols = 0;
    int tmp_num_sols;
    std::vector<ResultType> tmpres;
    std::vector<JunctionMatch> jmatches;

    // junc0 - junc1
    tmpres.clear();
    jmatches.clear();
    jmatches.push_back(junction_matches[0]);
    jmatches.push_back(junction_matches[1]);
    tmp_num_sols = RelativePoseSolver2vp2pt::MinimalSolver(new_vp_matches, jmatches, &tmpres);
    res->insert(res->end(), tmpres.begin(), tmpres.end());
    num_sols += tmp_num_sols;

    // junc0 - junc2
    tmpres.clear();
    jmatches.clear();
    jmatches.push_back(junction_matches[0]);
    jmatches.push_back(junction_matches[2]);
    tmp_num_sols = RelativePoseSolver2vp2pt::MinimalSolver(new_vp_matches, jmatches, &tmpres);
    res->insert(res->end(), tmpres.begin(), tmpres.end());
    num_sols += tmp_num_sols;

    // junc1 - junc2
    tmpres.clear();
    jmatches.clear();
    jmatches.push_back(junction_matches[1]);
    jmatches.push_back(junction_matches[2]);
    tmp_num_sols = RelativePoseSolver2vp2pt::MinimalSolver(new_vp_matches, jmatches, &tmpres);
    res->insert(res->end(), tmpres.begin(), tmpres.end());
    num_sols += tmp_num_sols;
    return num_sols;
}

int RelativePoseSolver1vp3cll_orthogonal::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
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

