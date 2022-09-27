#include "estimators/relative_pose_solver_homography.h"
#include "solvers/solver_H.h"
#include "solvers/5pt_solver.hxx"

namespace line_relative_pose {

int RelativePoseSolverHomography::DecomposeHomography(const std::vector<M3D>& Hs, std::vector<ResultType>* res) const {
    res->clear();
    for (size_t i = 0; i < Hs.size(); ++i) {
        M3D Es[10];
        int num_sols = solve_homography_calibrated(Hs[i], Es);
        for (size_t j = 0; j < num_sols; ++j) {
            M3D Rs[4];
            V3D ts[4];
            decompose_essential(Es[j], Rs, ts);
            for (size_t k = 0; k < 4; ++k) {
                res->push_back(std::make_pair(Rs[k], ts[k]));
            }
        }
    }
    return static_cast<int>(res->size());
}

int RelativePoseSolverHomography::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<PointMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    std::vector<int> min_sample_sizes = min_sample_size();
    THROW_CHECK_EQ(line_matches.size(), min_sample_sizes[0]);
    THROW_CHECK_EQ(vp_matches.size(), min_sample_sizes[1]);
    THROW_CHECK_EQ(junction_matches.size(), min_sample_sizes[2]);

    std::vector<M3D> Hs;
    int num_H = HomographySolver(line_matches, vp_matches, junction_matches, &Hs);
    return DecomposeHomography(Hs, res);
}

}  // namespace line_relative_pose 

