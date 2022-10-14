#include "estimators/relative_pose_solver_1line_1vp_2pt_orthogonal.h"

namespace line_relative_pose {

int RelativePoseSolver1line1vp2pt_orthogonal::MinimalSolver(const std::vector<LineMatch>& line_matches,
                                                            const std::vector<VPMatch>& vp_matches,
                                                            const std::vector<JunctionMatch>& junction_matches,
                                                            std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 1);
    THROW_CHECK_EQ(vp_matches.size(), 1);
    THROW_CHECK_EQ(junction_matches.size(), 2);

    std::vector<VPMatch> new_vp_matches;
    new_vp_matches.push_back(vp_matches[0]);

    // generate the other vp match from orthogonality
    V3D vq1 = line_matches[0].first.coords().cross(vp_matches[0].first).normalized();
    V3D vq2 = line_matches[0].second.coords().cross(vp_matches[0].second).normalized();
    new_vp_matches.push_back(std::make_pair(vq1, vq2));
    return RelativePoseSolver2vp2pt::MinimalSolver(new_vp_matches, junction_matches, res);
}

int RelativePoseSolver1line1vp2pt_orthogonal::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    THROW_CHECK_EQ(line_matches.size(), 1);
    THROW_CHECK_EQ(vp_matches.size(), 1);
    THROW_CHECK_EQ(junction_matches.size(), 2);

    return MinimalSolver(line_matches, vp_matches, junction_matches, res);
}

}  // namespace line_relative_pose 

