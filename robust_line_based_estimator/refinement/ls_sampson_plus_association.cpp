#include "base/pose.h"
#include "refinement/ls_sampson.h"
#include "refinement/cost_functions.h"

#include <ceres/ceres.h>

namespace line_relative_pose {


void LeastSquares_Sampson_plus_Association(
    const std::vector<JunctionMatch>& junction_matches, 
    const std::vector<LineMatch>& line_matches,
    const std::vector<std::pair<int, int>>& associations,
    std::tuple<M3D, V3D, M3D>* res)
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;

    // Set up problem
    V4D qvec = RotationMatrixToQuaternion(std::get<0>(*res));
    V3D tvec = std::get<1>(*res);
    ceres::Problem problem;

    // add sampson error
    for (auto it = junction_matches.begin(); it != junction_matches.end(); ++it) {
        V3D p1 = homogeneous(it->first.point());
        V3D p2 = homogeneous(it->second.point());
        ceres::LossFunction* loss_function = new ceres::TrivialLoss();
        ceres::CostFunction* cost_function = SampsonCostFunctor::Create(p1, p2);
        problem.AddResidualBlock(cost_function, loss_function, qvec.data(), tvec.data());
    }

    // add association error
    for (auto it = associations.begin(); it != associations.end(); ++it) {
        limap::Line2d l1 = line_matches[it->first].first;
        limap::Line2d l2 = line_matches[it->first].second;
        V2D p1 = junction_matches[it->second].first.point();
        V2D p2 = junction_matches[it->second].second.point();
        
        // check degeneracy
        const double degeneracy_angle_threshold = 10.0;
        M3D E = std::get<2>(*res);
        V3D epline1 = E.transpose() * homogeneous(p2);
        double angle1 = acos(std::abs(l1.direction().dot(V2D(epline1[2], -epline1[1]).normalized()))) * 180.0 / M_PI; 
        if (angle1 < degeneracy_angle_threshold)
            continue;
        V3D epline2 = E * homogeneous(p1);
        double angle2 = acos(std::abs(l2.direction().dot(V2D(epline2[2], -epline2[1]).normalized()))) * 180.0 / M_PI; 
        if (angle2 < degeneracy_angle_threshold)
            continue;

        ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0); // TODO: maybe we need to tune this
        const double loss_weight = 1.0; // TODO: maybe we need to tune this as well
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, loss_weight, ceres::DO_NOT_TAKE_OWNERSHIP);
        ceres::CostFunction* cost_function = AssociationCostFunctor::Create(p1, p2, l1, l2);
        problem.AddResidualBlock(cost_function, scaled_loss_function, qvec.data(), tvec.data());
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

// This parameterization seems unnecessary empirically
//     if (problem.HasParameterBlock(tvec.data())) {
// #ifdef CERES_PARAMETERIZATION_ENABLED
//         ceres::LocalParameterization* homo3d_parameterization = 
//             new ceres::HomogeneousVectorParameterization(3);
//         problem.SetParameterization(tvec.data(), homo3d_parameterization);
// #else
//         ceres::Manifold* homo3d_manifold = 
//             new ceres::SphereManifold<3>;
//         problem.SetManifold(tvec.data(), homo3d_manifold);
// #endif
//     }

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);
    std::get<0>(*res) = QuaternionToRotationMatrix(qvec);
    std::get<1>(*res) = tvec;
}

} // namespace line_relative_pose 

