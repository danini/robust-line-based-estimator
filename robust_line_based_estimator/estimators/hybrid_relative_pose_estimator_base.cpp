#include "estimators/hybrid_relative_pose_estimator_base.h"
#include "refinement/ls_sampson.h"
#include "refinement/ls_combined.h"
#include <iostream>
#include <chrono>

namespace line_relative_pose {

HybridRelativePoseEstimatorBase::HybridRelativePoseEstimatorBase(
            const M3D& K1, const M3D& K2,
            const std::pair<Eigen::Matrix4Xd, Eigen::Matrix4Xd>& line_matches,
            const std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>& vp_matches,
            const std::pair<std::vector<Junction2d>, std::vector<Junction2d>>& junction_matches,
            const std::pair<std::vector<int>, std::vector<int>>& vp_labels,
            const int ls_refinement, 
            const std::vector<double>& weights_refinement) 
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
        m_norm_lines_.push_back(normalize_line_match(m_lines_.back()));        
    }

    // initiate vp matches
    THROW_CHECK_EQ(vp_matches.first.cols(), vp_matches.second.cols());
    int num_m_vps = vp_matches.first.cols();
    for (size_t i = 0; i < num_m_vps; ++i) {
        V3D vp1 = vp_matches.first.col(i);
        V3D vp2 = vp_matches.second.col(i);
        m_vps_.push_back(std::make_pair(vp1, vp2));
        m_norm_vps_.push_back(normalize_vp_match(m_vps_.back()));
    }

    // initiate junction matches
    THROW_CHECK_EQ(junction_matches.first.size(), junction_matches.second.size());
    int num_m_junctions = junction_matches.first.size();
    for (size_t i = 0; i < num_m_junctions; ++i) {
        Junction2d p1 = junction_matches.first[i];
        Junction2d p2 = junction_matches.second[i];
        m_junctions_.push_back(std::make_pair(p1, p2));
        m_norm_junctions_.push_back(normalize_junction_match(m_junctions_.back()));
    }

    // initiate vp labels
    vp_to_line_ids_img1_.resize(num_m_vps);
    for (int i = 0; i < vp_labels.first.size(); ++i) {
        if (vp_labels.first[i] < 0)
            continue;
        int label = vp_labels.first[i];
        THROW_CHECK_LT(label, num_m_vps);
        vp_to_line_ids_img1_[label].push_back(i);
    }
    vp_to_line_ids_img2_.resize(num_m_vps);
    for (int i = 0; i < vp_labels.second.size(); ++i) {
        if (vp_labels.second[i] < 0)
            continue;
        int label = vp_labels.second[i];
        THROW_CHECK_LT(label, num_m_vps);
        vp_to_line_ids_img2_[label].push_back(i);
    }

    // init options
    ls_refinement_ = ls_refinement;
    THROW_CHECK_EQ(weights_refinement.size(), 2)
    weights_refinement_ = weights_refinement;
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
    V3D C2 = -T.transpose() * R;
    V3D n1e; n1e << p1(0), p1(1), 1;
    V3D n2e; n2e << p2(0), p2(1), 1;
    n2e = n2e.transpose() * R;
    const double a11 = n1e(0) * n1e(0) + n1e(1) * n1e(1) + n1e(2) * n1e(2),
        a12_21 = -(n1e(0) * n2e(0) + n1e(1) * n2e(1) + n1e(2) * n2e(2)),
        a22 = n2e(0) * n2e(0) + n2e(1) * n2e(1) + n2e(2) * n2e(2);
    M2D A; A << a11, a12_21, a12_21, a22;
    V2D b; b(0) = n1e.dot(C2); b(1) = n2e.dot(-C2);
    M2D Ainv; Ainv << A(1,1), -A(0,1), -A(1,0), A(0,0);
    Ainv *= 1.0 / (A(0,0) * A(1,1) - A(0,1) * A(1,0));
    V2D res = Ainv * b;
    return res[0] > 0.0 && res[1] > 0.0;
}

double HybridRelativePoseEstimatorBase::EvaluateModelOnVP(const ResultType& model, int i) const {
    const V3D &vp1 = m_norm_vps_[i].first;
    const V3D &vp2 = m_norm_vps_[i].second;
    const M3D& R = std::get<0>(model);
    
    V3D vp1_rotated = R * vp1;
    double cos_val = std::abs(vp1_rotated.dot(vp2));
    return acos(cos_val) * 180.0 / M_PI;
}

double HybridRelativePoseEstimatorBase::EvaluateModelOnJunction(const ResultType& model, int i) const {
    const V2D &p1 = m_norm_junctions_[i].first.point();
    const V2D &p2 = m_norm_junctions_[i].second.point();

    // fundamental matrix
    const M3D &R = std::get<0>(model); 
    const V3D &T = std::get<1>(model); 
    if (!check_cheirality(p1, p2, R, T))
        return std::numeric_limits<double>::max();

    const M3D &E = std::get<2>(model);

    const double 
        &x1 = p1(0),
        &y1 = p1(1),
        &x2 = p2(0),
        &y2 = p2(1);

    const double 
        &e11 = E(0, 0),
        &e12 = E(0, 1),
        &e13 = E(0, 2),
        &e21 = E(1, 0),
        &e22 = E(1, 1),
        &e23 = E(1, 2),
        &e31 = E(2, 0),
        &e32 = E(2, 1),
        &e33 = E(2, 2);

    double rxc = e11 * x2 + e21 * y2 + e31;
    double ryc = e12 * x2 + e22 * y2 + e32;
    double rwc = e13 * x2 + e23 * y2 + e33;
    double r = (x1 * rxc + y1 * ryc + rwc);
    double rx = e11 * x1 + e12 * y1 + e13;
    double ry = e21 * x1 + e22 * y1 + e23;

    return sqrt(r * r /
        (rxc * rxc + ryc * ryc + rx * rx + ry * ry));
}

double HybridRelativePoseEstimatorBase::EvaluateModelOnPoint(const ResultType& model, int t, int i) const {
    // Now that all lines are considered to be inliers
    if (t == 0) {
        return 0.0;
    }
    else if (t == 1) {
        return EvaluateModelOnVP(model, i);
    }
    else if (t == 2) {
        return EvaluateModelOnJunction(model, i);
    }
    else
        throw std::runtime_error("Error! Not supported.");
}

void HybridRelativePoseEstimatorBase::LeastSquares(const std::vector<std::vector<int>>& sample, ResultType* res) const {
    std::vector<JunctionMatch> junction_matches;
    for (size_t i = 0; i < sample[2].size(); ++i) {
        junction_matches.push_back(m_norm_junctions_[sample[2][i]]);
    } 
    if (ls_refinement_ == 0) {
        LeastSquares_Sampson(junction_matches, res);
    }
    else if (ls_refinement_ == 1 || ls_refinement_ == 2) {
        // no vp as inliers
        if (sample[1].empty())
            LeastSquares_Sampson(junction_matches, res);

        // here we start
        std::vector<VPMatch> vp_matches;
        std::vector<std::vector<limap::Line2d>> sup_lines_img1; // supporting lines
        std::vector<std::vector<limap::Line2d>> sup_lines_img2;
        for (size_t i = 0; i < sample[1].size(); ++i) {
            int vp_id = sample[1][i];
            // add vp match
            vp_matches.push_back(m_norm_vps_[vp_id]);
            // add supporting lines for image 1
            std::vector<limap::Line2d> lines_img1;
            for (size_t j = 0; j < vp_to_line_ids_img1_[vp_id].size(); ++j) {
                lines_img1.push_back(m_norm_lines_[vp_to_line_ids_img1_[vp_id][j]].first);
            }
            sup_lines_img1.push_back(lines_img1);
            // add supporting lines for image 2
            std::vector<limap::Line2d> lines_img2;
            for (size_t j = 0; j < vp_to_line_ids_img2_[vp_id].size(); ++j) {
                lines_img2.push_back(m_norm_lines_[vp_to_line_ids_img2_[vp_id][j]].second);
            }
            sup_lines_img2.push_back(lines_img2);
        }
        const double w_vp = weights_refinement_[0];
        const double w_line_vp = weights_refinement_[1];
        if (ls_refinement_ == 1)
            LeastSquares_Combined(junction_matches, vp_matches, sup_lines_img1, sup_lines_img2, res, w_vp, w_line_vp, false);
        else
            LeastSquares_Combined(junction_matches, vp_matches, sup_lines_img1, sup_lines_img2, res, w_vp, w_line_vp, true);
    }
    else {
        throw std::runtime_error("Error! Not supported");
    }
    // update E
    std::get<2>(*res) = essential_from_rel_pose(
        std::get<0>(*res), 
        std::get<1>(*res));
}

} // namespace line_relative_pose

