#include "estimators/relative_pose_solver_5pt.h"
#include "solvers/5pt_solver.hxx"

namespace line_relative_pose {

int RelativePoseSolver5pt::MinimalSolver(const std::vector<JunctionMatch>& junction_matches,
                                         std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(junction_matches.size(), 5);
    std::vector<V2D> points1(5), points2(5);
    for (size_t i = 0; i < 5; ++i) {
        points1[i] = junction_matches[i].first.point();
        points2[i] = junction_matches[i].second.point();
    }

    // solve for essential matrix
    M3D Es[10];
    int num_essentials = essential_solver(points1.data(), points2.data(), Es);

    // decompose essential matrix one by one
    res->resize(4 * num_essentials);
    for (size_t i = 0; i < num_essentials; ++i) {
        M3D Rs[4]; V3D ts[4];
        int num_decomposed = decompose_essential(Es[i], Rs, ts);
        THROW_CHECK_EQ(num_decomposed, 4);
        (*res)[4*i] = std::make_tuple(Rs[0], ts[0], Es[i]);
        (*res)[4*i + 1] = std::make_tuple(Rs[1], ts[1], Es[i]);
        (*res)[4*i + 2] = std::make_tuple(Rs[2], ts[2], Es[i]);
        (*res)[4*i + 3] = std::make_tuple(Rs[3], ts[3], Es[i]);
    }
    return res->size();
}

int RelativePoseSolver5pt::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 0);
    THROW_CHECK_EQ(vp_matches.size(), 0);
    THROW_CHECK_EQ(junction_matches.size(), 5);

    return MinimalSolver(junction_matches, res);
}

}  // namespace line_relative_pose 

