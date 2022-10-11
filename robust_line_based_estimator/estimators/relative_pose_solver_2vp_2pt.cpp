#include "estimators/relative_pose_solver_2vp_2pt.h"
#include "solvers/solver_2vp_2pt.h"

namespace line_relative_pose {

int RelativePoseSolver2vp2pt::MinimalSolver(const std::vector<VPMatch>& vp_matches,
                                            const std::vector<JunctionMatch>& junction_matches,
                                            std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(vp_matches.size(), 2);
    THROW_CHECK_EQ(junction_matches.size(), 2);

    M3D Rs[4];
    int num_sols = stage_1_solver_rotation_2vp(vp_matches[0].first, vp_matches[0].second,
                                               vp_matches[1].first, vp_matches[1].second, Rs);
    res->resize(num_sols);
    for (size_t i = 0; i < num_sols; ++i) {
        V3D t;
        stage_2_solver_translation_2pt(homogeneous(junction_matches[0].first.point()),
                                       homogeneous(junction_matches[0].second.point()),
                                       homogeneous(junction_matches[1].first.point()),
                                       homogeneous(junction_matches[1].second.point()),
                                       Rs[i], t);
        (*res)[i] = std::make_tuple(Rs[i], t, M3D());
    }
    return num_sols;
}

int RelativePoseSolver2vp2pt::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 0);
    THROW_CHECK_EQ(vp_matches.size(), 2);
    THROW_CHECK_EQ(junction_matches.size(), 2);

    return MinimalSolver(vp_matches, junction_matches, res);
}

}  // namespace line_relative_pose 

