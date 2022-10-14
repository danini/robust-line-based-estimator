#include "estimators/hybrid_relative_pose_estimator.h"

#include "estimators/relative_pose_solver_base.h"
#include "estimators/relative_pose_solver_5pt.h"
#include "estimators/relative_pose_solver_4line.h"
#include "estimators/relative_pose_solver_1vp_3pt.h"
#include "estimators/relative_pose_solver_1vp_3cll.h"
#include "estimators/relative_pose_solver_2vp_2pt.h"
#include "estimators/relative_pose_solver_2vp_3cll.h"
#include "estimators/relative_pose_solver_1line_1vp_2pt_orthogonal.h"
#include "estimators/relative_pose_solver_1vp_2pt_orthogonal.h"
#include "estimators/relative_pose_solver_1vp_3cll_orthogonal.h"

namespace line_relative_pose {

void HybridRelativePoseEstimator::InitSolvers() {
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver5pt()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver4line()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver1vp3pt()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver2vp2pt()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver1vp3cll()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver2vp3cll()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver1line1vp2pt_orthogonal()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver1vp2pt_orthogonal()));
    AddSolver(std::shared_ptr<RelativePoseSolverBase>(new RelativePoseSolver1vp3cll_orthogonal()));
}

} // namespace line_relative_pose

