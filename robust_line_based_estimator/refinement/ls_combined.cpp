#include "base/pose.h"
#include "refinement/ls_combined.h"
#include "refinement/cost_functions.h"

#include <ceres/ceres.h>

namespace line_relative_pose {

void LeastSquares_Combined(const std::vector<JunctionMatch>& junction_matches, 
                           const std::vector<VPMatch>& vp_matches,
                           const std::vector<std::vector<limap::Line2d>>& sup_lines_img1,
                           const std::vector<std::vector<limap::Line2d>>& sup_lines_img2,
                           std::tuple<M3D, V3D, M3D>* res,
                           double weight_vp_rotation,
                           double weight_line_vp,
                           bool fix_vp)
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;

    // Set up problem
    ceres::Problem problem;
    V4D qvec = RotationMatrixToQuaternion(std::get<0>(*res));
    V3D tvec = std::get<1>(*res);
    std::vector<VPMatch> vp_matches_data = vp_matches;
    int n_vps = vp_matches_data.size(); // number of vps to be optimized

    // add sampson residuals
    for (auto it = junction_matches.begin(); it != junction_matches.end(); ++it) {
        V3D p1 = homogeneous(it->first.point());
        V3D p2 = homogeneous(it->second.point());
        ceres::LossFunction* loss_function = new ceres::TrivialLoss();
        ceres::CostFunction* cost_function = SampsonCostFunctor::Create(p1, p2);
        problem.AddResidualBlock(cost_function, loss_function, qvec.data(), tvec.data());
    }

    // add vp residuals
    for (auto it = vp_matches_data.begin(); it != vp_matches_data.end(); ++it) {
        ceres::LossFunction* loss_function = new ceres::TrivialLoss(); // TODO: tune this
        ceres::CostFunction* cost_function = VpRotationCostFunctor::Create();
        double weight = weight_vp_rotation;
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
        problem.AddResidualBlock(cost_function, scaled_loss_function, qvec.data(), it->first.data(), it->second.data());
    }

    // add line-vp residuals
    if (!fix_vp) {
        ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.5); // TODO: tune this

        // compute sum of line length
        std::vector<double> sum_length_sup_lines_img1, sum_length_sup_lines_img2;
        sum_length_sup_lines_img1.resize(n_vps);
        std::fill(sum_length_sup_lines_img1.begin(), sum_length_sup_lines_img1.end(), 0.0);
        sum_length_sup_lines_img2.resize(n_vps);
        std::fill(sum_length_sup_lines_img2.begin(), sum_length_sup_lines_img2.end(), 0.0);
        for (size_t vp_id = 0; vp_id < n_vps; ++vp_id) {
            for (const limap::Line2d& line: sup_lines_img1[vp_id])
                sum_length_sup_lines_img1[vp_id] += line.length();
            for (const limap::Line2d& line: sup_lines_img2[vp_id])
                sum_length_sup_lines_img2[vp_id] += line.length();
        }

        for (size_t vp_id = 0; vp_id < n_vps; ++vp_id) {
            // add residuals from image 1
            for (const limap::Line2d& line: sup_lines_img1[vp_id]) {
                V2D midpoint = line.midpoint();
                ceres::CostFunction* cost_function = VpCostFunctor::Create(line.start[0], line.start[1], midpoint[0], midpoint[1]);
                double weight = weight_line_vp * line.length() / sum_length_sup_lines_img1[vp_id];
                ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost_function, scaled_loss_function, vp_matches_data[vp_id].first.data());
            }
            // add residuals from image 2
            for (const limap::Line2d& line: sup_lines_img2[vp_id]) {
                V2D midpoint = line.midpoint();
                ceres::CostFunction* cost_function = VpCostFunctor::Create(line.start[0], line.start[1], midpoint[0], midpoint[1]);
                double weight = weight_line_vp * line.length() / sum_length_sup_lines_img2[vp_id];
                ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost_function, scaled_loss_function, vp_matches_data[vp_id].second.data());
            }
        }
    }

// parameterize
    if (problem.HasParameterBlock(qvec.data())) {
#ifdef CERES_PARAMETERIZATION_ENABLED
        ceres::LocalParameterization* quaternion_parameterization = 
            new ceres::QuaternionParameterization;
        problem.SetParameterization(qvec.data(), quaternion_parameterization);
#else
        ceres::Manifold* quaternion_manifold = 
            new ceres::QuaternionManifold;
        problem.SetManifold(qvec.data(), quaternion_manifold);
#endif
    }

    for (size_t vp_id = 0; vp_id < n_vps; ++vp_id) {
        // parameterize vp1
        double* vp_ptr1 = vp_matches_data[vp_id].first.data();
        if (problem.HasParameterBlock(vp_ptr1)) {
            if (fix_vp)
                problem.SetParameterBlockConstant(vp_ptr1);
            else {
#ifdef CERES_PARAMETERIZATION_ENABLED
                ceres::LocalParameterization* homo3d_parameterization = 
                    new ceres::HomogeneousVectorParameterization(3);
                problem.SetParameterization(vp_ptr1, homo3d_parameterization);
#else
                ceres::Manifold* homo3d_manifold = 
                    new ceres::SphereManifold<3>;
                problem.SetManifold(vp_ptr1, homo3d_manifold);
#endif
            }
        }
        // parameterize vp2
        double* vp_ptr2 = vp_matches_data[vp_id].second.data();
        if (problem.HasParameterBlock(vp_ptr2)) {
            if (fix_vp)
                problem.SetParameterBlockConstant(vp_ptr2);
            else {
#ifdef CERES_PARAMETERIZATION_ENABLED
                ceres::LocalParameterization* homo3d_parameterization = 
                    new ceres::HomogeneousVectorParameterization(3);
                problem.SetParameterization(vp_ptr2, homo3d_parameterization);
#else
                ceres::Manifold* homo3d_manifold = 
                    new ceres::SphereManifold<3>;
                problem.SetManifold(vp_ptr2, homo3d_manifold);
#endif
            }
        }
    }

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);
    // std::cout<<"[LOG] "<<summary.BriefReport()<<std::endl;
    std::get<0>(*res) = QuaternionToRotationMatrix(qvec);
    std::get<1>(*res) = tvec;
}

} // namespace line_relative_pose 

