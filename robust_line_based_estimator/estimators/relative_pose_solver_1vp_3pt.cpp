#include "estimators/relative_pose_solver_1vp_3pt.h"
#include "solvers/solver_1vp_3pt.h"

namespace line_relative_pose {

int RelativePoseSolver1vp3pt::MinimalSolver(const std::vector<VPMatch>& vp_matches,
                                            const std::vector<JunctionMatch>& junction_matches,
                                            std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(vp_matches.size(), 1);
    THROW_CHECK_EQ(junction_matches.size(), 3);

    V3D vp = vp_matches[0].first;
    V3D vq = vp_matches[0].second;
    V3D pts[3], qts[3];
    for (size_t i = 0; i < 3; ++i) {
        pts[i] = homogeneous(junction_matches[i].first.point());
        qts[i] = homogeneous(junction_matches[i].second.point());
    }
    M3D Rs[12]; V3D ts[12];
    int num_sols = solver_wrapper_1vp_3pt(vp, vq, pts, qts, Rs, ts);
    res->resize(num_sols);
    for (size_t i = 0; i < num_sols; ++i) {
        (*res)[i] = std::make_tuple(Rs[i], ts[i], M3D());
    }
    return num_sols;
}

int RelativePoseSolver1vp3pt::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 0);
    THROW_CHECK_EQ(vp_matches.size(), 1);
    THROW_CHECK_EQ(junction_matches.size(), 3);

    return MinimalSolver(vp_matches, junction_matches, res);
}

}  // namespace line_relative_pose 

