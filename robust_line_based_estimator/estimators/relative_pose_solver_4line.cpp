#include "estimators/relative_pose_solver_4line.h"

namespace line_relative_pose {

int RelativePoseSolver4line::HomographySolver(const std::vector<LineMatch>& line_matches,
                                              const std::vector<VPMatch>& vp_matches,
                                              const std::vector<PointMatch>& junction_matches,
                                              std::vector<M3D>* Hs) const 
{
    THROW_CHECK_EQ(line_matches.size(), 4);

    Eigen::Matrix<double, Eigen::Dynamic, 9>  A(8, 9);
    for (size_t i = 0; i < 4; ++i) {
        // p1(1:2) * (h3' * p2) = p1(3) * [h1';h2'] * p2         
        // [p1(3)*p2' 0 0 0 -p1(1)*p2';
        //  0 0 0 p1(3)*p2' -p1(2)*p2'] * [h1;h2;h3] = 0

        V3D l1 = line_matches[i].first.coords();
        V3D l2 = line_matches[i].second.coords();
        A.row(2*i  ) << l1(2) * l2.transpose(), 0.0, 0.0, 0.0, -l1(0)*l2.transpose();
        A.row(2*i+1) << 0.0, 0.0, 0.0, l1(2) * l2.transpose(), -l1(1)*l2.transpose();
    }

    Eigen::JacobiSVD<decltype(A)> svd(A, Eigen::ComputeFullV);

    Eigen::Matrix<double,9,1> h = svd.matrixV().rightCols<1>();

    Eigen::Matrix3d H;
    H.row(0) = h.block<3,1>(0,0).transpose();
    H.row(1) = h.block<3,1>(3,0).transpose();
    H.row(2) = h.block<3,1>(6,0).transpose();

    Hs->resize(1);
    (*Hs)[0] = H;
    return 1;
}

}  // namespace line_relative_pose 

