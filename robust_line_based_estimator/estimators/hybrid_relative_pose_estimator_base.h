#ifndef LINE_RELATIVE_POSE_ESTIMATORS_HYBRID_RELATIVE_POSE_ESTIMATOR_BASE_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_HYBRID_RELATIVE_POSE_ESTIMATOR_BASE_H_ 

#include "base/types.h"
#include "base/junction.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace line_relative_pose {

// Only supporting calibrated case
class HybridRelativePoseEstimatorBase {
public:
    // Input: K1, K2 and un-normalized data
    HybridRelativePoseEstimatorBase(const M3D& K1, const M3D& K2,
                                    const std::pair<Eigen::Matrix4Xd, Eigen::Matrix4Xd>& line_matches,
                                    const std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>& vp_matches,
                                    const std::pair<std::vector<Junction2d>, std::vector<Junction2d>>& junction_matches,
                                    const std::pair<std::vector<int>, std::vector<int>>& vp_labels,
                                    int ls_refinement,
                                    const std::vector<double>& weights_refinement);

    using ResultType = std::tuple<M3D, V3D, M3D>;

    inline int num_data_types() const { return 3; }

    inline void num_data(std::vector<int>* num_data) const {
        num_data->resize(3);
        (*num_data)[0] = m_lines_.size();
        (*num_data)[1] = m_vps_.size();
        (*num_data)[2] = m_junctions_.size();
    }

    // Set prior probabilities
    void set_prior_probabilities(const std::vector<double>& solver_probabilities);

    // Get prior probabilities
    void solver_probabilities(std::vector<double>* solver_probabilities) const;

    // Evaluates the line on the i-th data point of the t-th data type.
    double EvaluateModelOnPoint(const ResultType& model, int t, int i, double *weight = nullptr) const;
  
    // Linear least squares solver. Calls NonMinimalSolver.
    void LeastSquares(const std::vector<std::vector<int>>& sample,
                      ResultType* res) const;

    void set_line_inlier_ratio(const double inlier_ratio) { line_inlier_ratio_ = inlier_ratio; }

    virtual inline int num_minimal_solvers() const = 0;

    virtual inline void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const = 0;

    virtual inline int MinimalSolver(const std::vector<std::vector<int>>& sample,
                                     const int solver_idx,
                                     std::vector<ResultType>* res) const = 0;

protected:
    // options
    int ls_refinement_ = 0; // 0 for sampson, 1 for sampson + vp + line, 2 for sampson + vp (fixed)
    std::vector<double> weights_refinement_; // 0 for vp rotation error, 1 for line-vp error
    double line_inlier_ratio_ = 0.3;

    // calibration
    M3D K1_inv_, K2_inv_;

    // normalized data
    std::vector<LineMatch> m_lines_,
        m_norm_lines_;
    std::vector<VPMatch> m_vps_,
        m_norm_vps_;
    std::vector<JunctionMatch> m_junctions_,
        m_norm_junctions_;

    // used for advanced sampling
    std::vector<std::vector<int>> vp_to_line_ids_img1_; // line ids for each vp
    std::vector<std::vector<int>> vp_to_line_ids_img2_; // line ids for each vp

    // set the entry to zero to disable particular solvers
    std::vector<double> prior_probabilities_;

    // scoring
    double EvaluateModelOnVP(const ResultType& model, int i) const;
    double EvaluateModelOnJunction(const ResultType& model, int i) const;

    // functions
    inline limap::Line2d normalize_line(const limap::Line2d& line, const M3D& K_inv) const { return limap::Line2d(normalize_point(line.start, K_inv), normalize_point(line.end, K_inv)); }
    inline V3D normalize_vp(const V3D& vp, const M3D& K_inv) const { return (K_inv * vp).normalized(); }
    inline V2D normalize_point(const V2D& p, const M3D& K_inv) const { return dehomogeneous(K_inv * homogeneous(p)); }
    inline Junction2d normalize_junction(const Junction2d& junc, const M3D& K_inv) const;
    inline LineMatch normalize_line_match(const LineMatch& line_match) const { return std::make_pair(normalize_line(line_match.first, K1_inv_), normalize_line(line_match.second, K2_inv_)); }
    inline VPMatch normalize_vp_match(const VPMatch& vp_match) const { return std::make_pair(normalize_vp(vp_match.first, K1_inv_), normalize_vp(vp_match.second, K2_inv_)); }
    inline JunctionMatch normalize_junction_match(const JunctionMatch& junction_match) const { return std::make_pair(normalize_junction(junction_match.first, K1_inv_), normalize_junction(junction_match.second, K2_inv_)); }
    inline M3D essential_from_rel_pose(const M3D &R, const V3D &T) const;

    bool check_cheirality(const V2D& p1, const V2D& p2, const M3D& R, const V3D& T) const;
};

inline M3D HybridRelativePoseEstimatorBase::essential_from_rel_pose(const M3D &R, const V3D &T) const
{    
    M3D tskew;
    tskew(0, 0) = 0.0; tskew(0, 1) = -T(2); tskew(0, 2) = T(1);
    tskew(1, 0) = T(2); tskew(1, 1) = 0.0; tskew(1, 2) = -T(0);
    tskew(2, 0) = -T(1); tskew(2, 1) = T(0); tskew(2, 2) = 0.0;
    return tskew * R;
}

inline Junction2d HybridRelativePoseEstimatorBase::normalize_junction(const Junction2d& junc, const M3D& K_inv) const {
    if (junc.IsJunction())
        return Junction2d(normalize_line(junc.line1(), K_inv), normalize_line(junc.line2(), K_inv));
    else
        return Junction2d(normalize_point(junc.point(), K_inv));
}

}  // namespace ransac_lib

#endif 

