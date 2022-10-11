#include "base/pose.h"
#include "refinement/ls_sampson.h"
#include "refinement/cost_functions.h"

#include <ceres/ceres.h>

namespace line_relative_pose {

void LeastSquares_Sampson(const std::vector<PointMatch>& junction_matches, std::pair<M3D, V3D>* res) 
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;

    // Set up problem
    V4D qvec = RotationMatrixToQuaternion(res->first);
    V3D tvec = res->second;
    ceres::Problem problem;
    for (auto it = junction_matches.begin(); it != junction_matches.end(); ++it) {
        V3D p1 = homogeneous(it->first);
        V3D p2 = homogeneous(it->second);
        ceres::LossFunction* loss_function = new ceres::TrivialLoss();
        ceres::CostFunction* cost_function = SampsonCostFunctor::Create(p1, p2);
        problem.AddResidualBlock(cost_function, loss_function, qvec.data(), tvec.data());
    }

    // parameterize
    if (problem.HasParameterBlock(qvec.data())) {
        ceres::LocalParameterization* quaternion_parameterization = 
            new ceres::QuaternionParameterization;
        problem.SetParameterization(qvec.data(), quaternion_parameterization);
    }

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);
    res->first = QuaternionToRotationMatrix(qvec);
    res->second = tvec;
}

} // namespace line_relative_pose 

