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

} // namespace line_relative_pose 

#endif

