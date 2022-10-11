#include "estimators/hybrid_relative_pose_estimator_base.h"
#include "refinement/ls_sampson.h"
#include <iostream>

namespace line_relative_pose {

HybridRelativePoseEstimatorBase::HybridRelativePoseEstimatorBase(
            const M3D& K1, const M3D& K2,
            const std::pair<Eigen::Matrix4Xd, Eigen::Matrix4Xd>& line_matches,
            const std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>& vp_matches,
            const std::pair<std::vector<Junction2d>, std::vector<Junction2d>>& junction_matches,
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
    THROW_CHECK_EQ(junction_matches.first.size(), junction_matches.second.size());
    int num_m_junctions = junction_matches.first.size();
    for (size_t i = 0; i < num_m_junctions; ++i) {
        Junction2d p1 = junction_matches.first[i];
        Junction2d p2 = junction_matches.second[i];
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

bool HybridRelativePoseEstimatorBase::check_cheirality(const V2D& p1, const V2D& p2, const M3D& R, const V3D& T) const {
    V3D C1 = V3D::Zero();
    V3D C2 = - R.transpose() * T;
    V3D n1e = homogeneous(p1);
    V3D n2e = R.transpose() * homogeneous(p2);
    M2D A; A << n1e.dot(n1e), -n1e.dot(n2e), -n2e.dot(n1e), n2e.dot(n2e);
    V2D b; b(0) = n1e.dot(C2 - C1); b(1) = n2e.dot(C1 - C2);
    V2D res = A.inverse() * b;
    return res[0] > 0.0 && res[1] > 0.0;
}

double HybridRelativePoseEstimatorBase::EvaluateModelOnPoint(const ResultType& model, int t, int i) const {
    // Now that all lines and vps are considered to be inliers
    if (t == 0 || t == 1) {
        return 0.0;
    }
    THROW_CHECK_EQ(t, 2);
    V2D p1 = m_junctions_[i].first.point();
    V2D p2 = m_junctions_[i].second.point();

    // fundamental matrix
    M3D R = model.first;
    V3D T = model.second;
    if (!check_cheirality(p1, p2, R, T))
        return std::numeric_limits<double>::max();

    // epipolar distance
    // TODO: test sampson distance
    // TODO: we shouldnt compute the fundamental matrix again and again
    M3D tskew;
    tskew(0, 0) = 0.0; tskew(0, 1) = -T(2); tskew(0, 2) = T(1);
    tskew(1, 0) = T(2); tskew(1, 1) = 0.0; tskew(1, 2) = -T(0);
    tskew(2, 0) = -T(1); tskew(2, 1) = T(0); tskew(2, 2) = 0.0;
    M3D E = tskew * R;
    M3D F = K2_inv_.transpose() * E * K1_inv_;
    V3D coor_epline1to2 = (F * homogeneous(p1)).normalized();
    double dist = std::abs(homogeneous(p2).dot(coor_epline1to2)) / V2D(coor_epline1to2(0), coor_epline1to2(1)).norm();
    return dist;
}

void HybridRelativePoseEstimatorBase::LeastSquares(const std::vector<std::vector<int>>& sample, ResultType* res) const {
    std::vector<JunctionMatch> junction_matches;
    for (size_t i = 0; i < sample[2].size(); ++i) {
        junction_matches.push_back(normalize_junction_match(m_junctions_[sample[2][i]]));
    } 
    LeastSquares_Sampson(junction_matches, res);
}

} // namespace line_relative_pose

