#include "estimators/hybrid_relative_pose_estimator_base.h"
#include <iostream>

namespace line_relative_pose {

HybridRelativePoseEstimatorBase::HybridRelativePoseEstimatorBase(
            const M3D& K1, const M3D& K2,
            const std::pair<Eigen::Matrix4Xd, Eigen::Matrix4Xd>& line_matches,
            const std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>& vp_matches,
            const std::pair<Eigen::Matrix2Xd, Eigen::Matrix2Xd>& junction_matches,
            const std::pair<std::vector<int>, std::vector<int>>& vp_labels) 
{
    // initiate calibrations
    K1_inv_ = K1.inverse(); 
    K2_inv_ = K2.inverse();

    // initiate line matches
    THROW_CHECK_EQ(line_matches.first.cols(), line_matches.second.cols());
    int num_m_lines = line_matches.first.cols();
    for (size_t i = 0; i < num_m_lines; ++i) {
        limap::Line2d line1 = limap::Line2d(Eigen::Vector4d(line_matches.first.col(i)));
        limap::Line2d line2 = limap::Line2d(Eigen::Vector4d(line_matches.second.col(i)));
        m_lines_.push_back(std::make_pair(line1, line2));
    }

    // initiate vp matches
    THROW_CHECK_EQ(vp_matches.first.cols(), vp_matches.second.cols());
    int num_m_vps = vp_matches.first.cols();
    for (size_t i = 0; i < num_m_vps; ++i) {
        V3D vp1 = vp_matches.first.col(i);
        V3D vp2 = vp_matches.second.col(i);
        m_vps_.push_back(std::make_pair(vp1, vp2));
    }

    // initiate junction matches
    THROW_CHECK_EQ(junction_matches.first.cols(), junction_matches.second.cols());
    int num_m_junctions = junction_matches.first.cols();
    for (size_t i = 0; i < num_m_junctions; ++i) {
        V2D p1 = junction_matches.first.col(i);
        V2D p2 = junction_matches.second.col(i);
        m_junctions_.push_back(std::make_pair(p1, p2));
    }

    // initiate vp labels
    vp_labels_img1_ = vp_labels.first;
    vp_labels_img2_ = vp_labels.second;
}

void HybridRelativePoseEstimatorBase::set_prior_probabilities(const std::vector<double>& probs) {
    THROW_CHECK_EQ(probs.size(), num_minimal_solvers());
    prior_probabilities_ = probs;
}

void HybridRelativePoseEstimatorBase::solver_probabilities(std::vector<double>* solver_probabilities) const {
    std::vector<double> probs = prior_probabilities_;
    if (probs.empty()) {
        probs.resize(num_minimal_solvers());
        std::fill(probs.begin(), probs.end(), 1.0 / num_minimal_solvers());
    }
    else {
        THROW_CHECK_EQ(probs.size(), num_minimal_solvers());
    }
    double sum_probabilities = 0;
    for (auto it = probs.begin(); it != probs.end(); ++it) {
        sum_probabilities += *it;
    }
    THROW_CHECK_GT(sum_probabilities, 0);
    *solver_probabilities = probs;
}

double HybridRelativePoseEstimatorBase::EvaluateModelOnPoint(const ResultType& model, int t, int i) const {
    // lines and vps do not contribute on the inlier counting
    if (t == 0 || t == 1) {
        return std::numeric_limits<double>::max();
    }
    THROW_CHECK_EQ(t, 2);

    // fundamental matrix
    // TODO: we shouldnt compute this again and again
    M3D R = model.first;
    V3D T = model.second;
    M3D tskew;
    tskew(0, 0) = 0.0; tskew(0, 1) = -T(2); tskew(0, 2) = T(1);
    tskew(1, 0) = T(2); tskew(1, 1) = 0.0; tskew(1, 2) = -T(0);
    tskew(2, 0) = -T(1); tskew(2, 1) = T(0); tskew(2, 2) = 0.0;
    M3D E = tskew * R;
    M3D F = K2_inv_.transpose() * E * K1_inv_;

    // epipolar distance
    V2D p1 = m_junctions_[i].first;
    V2D p2 = m_junctions_[i].second;
    V3D coor_epline1to2 = (F * homogeneous(p1)).normalized();
    double dist = std::abs(homogeneous(p2).dot(coor_epline1to2)) / V2D(coor_epline1to2(0), coor_epline1to2(1)).norm();
    return dist;
}

void HybridRelativePoseEstimatorBase::LeastSquares(const std::vector<std::vector<int>>& sample, ResultType* res) const {
    // TODO
    return;
}

} // namespace line_relative_pose

