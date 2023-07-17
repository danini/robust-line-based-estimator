#include "base/pose.h"
#include "refinement/ls_sampson.h"
#include "refinement/cost_functions.h"

#include <ceres/ceres.h>

namespace line_relative_pose {

void LeastSquares_Sampson(const std::vector<JunctionMatch>& junction_matches, std::tuple<M3D, V3D, M3D>* res) 
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
    for (auto it = junction_matches.begin(); it != junction_matches.end(); ++it) {
        V3D p1 = homogeneous(it->first.point());
        V3D p2 = homogeneous(it->second.point());
        ceres::LossFunction* loss_function = new ceres::TrivialLoss();
        ceres::CostFunction* cost_function = SampsonCostFunctor::Create(p1, p2);
        problem.AddResidualBlock(cost_function, loss_function, qvec.data(), tvec.data());
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
    if (problem.HasParameterBlock(tvec.data())) {
#ifdef CERES_PARAMETERIZATION_ENABLED
        ceres::LocalParameterization* homo3d_parameterization = 
            new ceres::HomogeneousVectorParameterization(3);
        problem.SetParameterization(tvec.data(), homo3d_parameterization);
#else
        ceres::Manifold* homo3d_manifold = 
            new ceres::SphereManifold<3>;
        problem.SetManifold(tvec.data(), homo3d_manifold);
#endif
    }

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);
    std::get<0>(*res) = QuaternionToRotationMatrix(qvec);
    std::get<1>(*res) = tvec;
}

} // namespace line_relative_pose 

