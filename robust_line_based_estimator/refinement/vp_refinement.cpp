#include "refinement/vp_refinement.h"
#include "refinement/cost_functions.h"

namespace line_relative_pose {

std::vector<V3D> VPRefinement(const std::vector<int>& labels,
    const std::vector<V4D>& inlier_lines, std::vector<V3D>& vps) {
    // Convert the lines
    std::vector<limap::Line2d> lines;
    for(V4D l: inlier_lines)
        lines.push_back(limap::Line2d(l));

    // Sum length (to get the final residual in pixel unit)
    size_t n_lines = lines.size();
    size_t n_vps = vps.size();
    double sum_length = 0;
    for (size_t i = 0; i < n_lines; ++i)
        sum_length += lines[i].length();

    // Optimize independently each VP with its own inliers using Ceres
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;

    for (size_t i = 0; i < n_vps; i++) {
        // Define the optimization problem
        ceres::Problem problem;
        for (size_t j = 0; j < n_lines; j++) {
            if (labels[j] != i)
                continue;

            // Mid point of the current line
            V2D midpoint = lines[j].midpoint();
            double weight = lines[j].length() / sum_length;
            ceres::LossFunction* vp_loss = new ceres::ScaledLoss(
                new ceres::CauchyLoss(0.5), weight, ceres::TAKE_OWNERSHIP);
            ceres::CostFunction* vp_cost_function = VpCostFunctor::Create(
                lines[j].start[0], lines[j].start[1], midpoint[0], midpoint[1]);
            problem.AddResidualBlock(vp_cost_function, vp_loss, vps[i].data());
        }
        if (problem.HasParameterBlock(vps[i].data())) {
            ceres::LocalParameterization* homo3d_parameterization = new ceres::HomogeneousVectorParameterization(3);
            problem.SetParameterization(vps[i].data(), homo3d_parameterization);
        }

        // Solve the optimization problem and update the VP
        ceres::Solve(options, &problem, &summary);
    }

    return vps;
}

} // namespace line_relative_pose 
