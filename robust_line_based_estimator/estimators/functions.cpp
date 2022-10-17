#include "estimators/functions.h"
#include "estimators/hybrid_relative_pose_estimator.h"
#include "estimators/line_relative_pose_hybrid_ransac.h"

namespace line_relative_pose {

// for each of the two input lines in line_matches: 4 x n array, with each column [x1, y1, x2, y2]
std::pair<std::tuple<M3D, V3D, M3D>, ransac_lib::HybridRansacStatistics> run_hybrid_relative_pose(
        const M3D& K1, const M3D& K2,
        const std::pair<Eigen::Matrix4Xd, Eigen::Matrix4Xd>& line_matches, 
        const std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>& vp_matches, 
        const std::pair<std::vector<Junction2d>, std::vector<Junction2d>>& junction_matches, 
        const std::pair<std::vector<int>, std::vector<int>> vp_labels,
        const ransac_lib::HybridLORansacOptions& options,
        const std::vector<bool>& solver_flags,
        const int ls_refinement,
        const std::vector<double>& weights_refinement) 
{
    ransac_lib::HybridLORansacOptions options_ = options;
    std::random_device rand_dev;
    options_.random_seed_ = rand_dev();

    // use valid solvers with solver_flags options
    HybridRelativePoseEstimator solver(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, ls_refinement, weights_refinement);
    std::vector<double> solver_probabilities;
    solver.solver_probabilities(&solver_probabilities);
    THROW_CHECK_EQ(solver_flags.size(), solver.num_minimal_solvers());
    for (size_t i = 0; i < solver_flags.size(); ++i) {
        if (!solver_flags[i])
            solver_probabilities[i] = 0;
    }
    solver.set_prior_probabilities(solver_probabilities);

    // hybrid ransac
    using ResultType = std::tuple<M3D, V3D, M3D>;
    HybridLineRelativePoseRansac<ResultType, std::vector<ResultType>, HybridRelativePoseEstimator> lomsac;
    ResultType best_model;
    ransac_lib::HybridRansacStatistics ransac_stats;
    int num_ransac_inliers = lomsac.EstimateModel(options_, solver, &best_model, &ransac_stats);
    return std::make_pair(best_model, ransac_stats);
}

} // namespace line_relative_pose 

