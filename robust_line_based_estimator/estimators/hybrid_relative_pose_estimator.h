#ifndef LINE_RELATIVE_POSE_ESTIMATORS_HYBRID_RELATIVE_POSE_ESTIMATOR_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_HYBRID_RELATIVE_POSE_ESTIMATOR_H_ 

#include "estimators/hybrid_relative_pose_estimator_base.h"
#include "estimators/relative_pose_solver_base.h"

#include <Eigen/Core>

namespace line_relative_pose {

class HybridRelativePoseEstimator: public HybridRelativePoseEstimatorBase {
public:
    // Input: K1, K2 and un-normalized data
    HybridRelativePoseEstimator(const M3D& K1, const M3D& K2,
                                const std::pair<Eigen::Matrix4Xd, Eigen::Matrix4Xd>& line_matches,
                                const std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>& vp_matches,
                                const std::pair<std::vector<Junction2d>, std::vector<Junction2d>>& junction_matches,
                                const std::pair<std::vector<int>, std::vector<int>>& vp_labels,
                                const int ls_refinement,
                                const std::vector<double>& weights_refinement):
        HybridRelativePoseEstimatorBase(K1, K2, line_matches, vp_matches, junction_matches, vp_labels, ls_refinement, weights_refinement) { InitSolvers(); }

    using ResultType = HybridRelativePoseEstimatorBase::ResultType;

    void InitSolvers();

    void AddSolver(const std::shared_ptr<RelativePoseSolverBase>& solver) { solvers_.push_back(solver); }

    inline int num_minimal_solvers() const { return solvers_.size(); }

    inline void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(solvers_.size());
        for (size_t i = 0; i < solvers_.size(); ++i) {
            (*min_sample_sizes)[i] = solvers_[i]->min_sample_size();
        }
    }

    inline int MinimalSolver(const std::vector<std::vector<int>>& sample,
                             const int solver_idx,
                             std::vector<ResultType>* res) const {
        THROW_CHECK_GE(solver_idx, 0);
        THROW_CHECK_LT(solver_idx, solvers_.size());

        // get data
        std::vector<LineMatch> line_matches;
        for (size_t i = 0; i < sample[0].size(); ++i) {
            line_matches.push_back(m_norm_lines_[sample[0][i]]);
        }
        std::vector<VPMatch> vp_matches;
        for (size_t i = 0; i < sample[1].size(); ++i) {
            vp_matches.push_back(m_norm_vps_[sample[1][i]]);
        }
        std::vector<JunctionMatch> junction_matches;
        for (size_t i = 0; i < sample[2].size(); ++i) {
            junction_matches.push_back(m_norm_junctions_[sample[2][i]]);
        }

        int num_solutions = solvers_[solver_idx]->MinimalSolverWrapper(line_matches, vp_matches, junction_matches, res);
        for (auto &[R, t, E] : *res)
            E = essential_from_rel_pose(R, t);
        return num_solutions;
    }

protected:
    std::vector<std::shared_ptr<RelativePoseSolverBase>> solvers_;
};

}  // namespace line_relative_pose 

#endif 

