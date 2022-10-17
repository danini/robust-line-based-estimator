#ifndef LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_COST_FUNCTIONS_H_
#define LINE_RELATIVE_POSE_ESTIMATORS_REFINEMENT_COST_FUNCTIONS_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace line_relative_pose {

struct SampsonCostFunctor {
    explicit SampsonCostFunctor(const V3D& p1, const V3D& p2): p1_(p1), p2_(p2) {}
  
    template <typename T>
    bool operator()(const T* qvec, const T* tvec, T* residuals) const {
        // compose essential matrix
        T R_pt[3 * 3];
        ceres::QuaternionToRotation(qvec, R_pt);
        Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>> R(R_pt);
        Eigen::Matrix<T, 3, 3> tskew;
        tskew(0, 0) = T(0.0); tskew(0, 1) = -tvec[2]; tskew(0, 2) = tvec[1];
        tskew(1, 0) = tvec[2]; tskew(1, 1) = T(0.0); tskew(1, 2) = -tvec[0];
        tskew(2, 0) = -tvec[1]; tskew(2, 1) = tvec[0]; tskew(2, 2) = T(0.0);
        Eigen::Matrix<T, 3, 3> E = tskew * R;

        // residual (sampson)
        Eigen::Matrix<T, 3, 1> p1, p2;
        p1[0] = T(p1_[0]); p1[1] = T(p1_[1]); p1[2] = T(p1_[2]);
        p2[0] = T(p2_[0]); p2[1] = T(p2_[1]); p2[2] = T(p2_[2]);
        T C = p2.transpose() * E * p1;
        Eigen::Matrix<T, 3, 1> epline1, epline2;
        epline1 = E.transpose() * p2;
        epline2 = E * p1;
        T nJc_sq = epline1[0] * epline1[0] + epline1[1] * epline1[1]
                   + epline2[0] * epline2[0] + epline2[1] * epline2[1];
        residuals[0] = C / ceres::sqrt(nJc_sq);
        return true;
    }
  
    static ceres::CostFunction* Create(const V3D& p1, const V3D& p2) {
        return new ceres::AutoDiffCostFunction<SampsonCostFunctor, 1, 4, 3>(new SampsonCostFunctor(p1, p2));
    }

private:
    const V3D p1_;
    const V3D p2_;
};

struct VpCostFunctor {
  explicit VpCostFunctor(
      double x1, double y1, double dcx, double dcy)
      : x1_(x1), y1_(y1), dcx_(dcx), dcy_(dcy) {}

  template <typename T>
  bool operator()(const T* vp, T* residuals) const {
    // Line passing through the VP and the center of the line segment
    // l = cross(dc, vp)
    T l1 = dcy_ * vp[2] - vp[1];
    T l2 = vp[0] - dcx_ * vp[2];
    T l3 = dcx_ * vp[1] - dcy_ * vp[0];

    // Residual = max orthogonal distance of the two endpoints to l
    residuals[0] = (ceres::abs(x1_ * l1 + y1_ * l2 + l3)
                    / ceres::sqrt(l1 * l1 + l2 * l2));

    return true;
  }

  static ceres::CostFunction* Create(double x1, double y1, double dcx, double dcy) {
    return new ceres::AutoDiffCostFunction<VpCostFunctor, 1, 3>(
        new VpCostFunctor(x1, y1, dcx, dcy));
  }

 private:
  const double x1_;
  const double y1_;
  const double dcx_;
  const double dcy_;
};

template <typename T>
T CeresComputeDist3D_sine(const T dir1[3], const T dir2[3]) {
    T dir1_norm = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] + dir1[2] * dir1[2] + EPS);
    T dir2_norm = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] + dir2[2] * dir2[2] + EPS);
    T dir1_normalized[3], dir2_normalized[3];
    for (size_t i = 0; i < 3; ++i) {
        dir1_normalized[i] = dir1[i] / dir1_norm;
        dir2_normalized[i] = dir2[i] / dir2_norm;
    }
    T res[3];
    ceres::CrossProduct(dir1_normalized, dir2_normalized, res);
    T sine = ceres::sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + EPS);
    if (sine > T(1.0))
        sine = T(1.0);
    return sine;
}

template <typename T>
T CeresComputeDist3D_cosine(const T dir1[3], const T dir2[3]) {
    T dir1_norm = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] + dir1[2] * dir1[2] + EPS);
    T dir2_norm = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] + dir2[2] * dir2[2] + EPS);
    T dir1_normalized[3], dir2_normalized[3];
    for (size_t i = 0; i < 3; ++i) {
        dir1_normalized[i] = dir1[i] / dir1_norm;
        dir2_normalized[i] = dir2[i] / dir2_norm;
    }
    T cosine = T(0.0);
    for (size_t i = 0; i < 3; ++i) {
        cosine += dir1_normalized[i] * dir2_normalized[i];
    }
    cosine = ceres::abs(cosine);
    if (cosine > T(1.0))
        cosine = T(1.0);
    return cosine;
}

struct VpRotationCostFunctor {
    explicit VpRotationCostFunctor() {}

    template <typename T>
    bool operator()(const T* qvec, const T* vp1, const T* vp2, T* residuals) const {
        // rotate vp1 
        T vp1_rotated[3];
        ceres::QuaternionRotatePoint(qvec, vp1, vp1_rotated);
        residuals[0] = CeresComputeDist3D_sine(vp1_rotated, vp2);
        return true;
    }

    static ceres::CostFunction* Create() {
        return new ceres::AutoDiffCostFunction<VpRotationCostFunctor, 1, 4, 3, 3>(new VpRotationCostFunctor());
    }
};

} // namespace line_relative_pose 

#endif

