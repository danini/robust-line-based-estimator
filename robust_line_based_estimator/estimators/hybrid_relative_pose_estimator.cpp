#include "estimators/hybrid_relative_pose_estimator.h"

#include "estimators/relative_pose_solver_base.h"
#include "estimators/relative_pose_solver_5pt.h"
#include "estimators/relative_pose_solver_4line.h"
#include "estimators/relative_pose_solver_1vp_3pt.h"
#include "estimators/relative_pose_solver_1vp_3cll.h"
#include "estimators/relative_pose_solver_2vp_2pt.h"
#include "estimators/relative_pose_solver_2vp_3cll.h"

namespace line_relative_pose {

void HybridRelativePoseEstimator::InitSolvers() {
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver5pt()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver4line()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver1vp3pt()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver1vp3cll()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver2vp2pt()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver2vp3cll()));
}

} // namespace line_relative_pose

