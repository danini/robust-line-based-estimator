#include "estimators/relative_pose_solver_homography.h"
#include "solvers/solver_H.h"
#include "solvers/5pt_solver.hxx"

namespace line_relative_pose {

int RelativePoseSolverHomography::DecomposeHomography(const std::vector<M3D>& Hs, std::vector<ResultType>* res) const {
    res->clear();
    for (size_t i = 0; i < Hs.size(); ++i) 
    {        
        // normalize H so that the second singular value is one
        Eigen::JacobiSVD<M3D> svd1(Hs[i]);
        M3D H2 = Hs[i] / svd1.singularValues()[1];

        // compute the SVD of the symmetric matrix H'*H = VSV'
        Eigen::JacobiSVD<M3D> svd2(H2.transpose() * H2,
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        M3D V = svd2.matrixV();

        // ensure V is right-handed
        if (V.determinant() < 0.0) V *= -1.0;

        // get the squared singular values
        V3D S = svd2.singularValues();
        double s1 = S[0];
        double s3 = S[2];

        V3D v1 = V.col(0);
        V3D v2 = V.col(1);
        V3D v3 = V.col(2);

        // pure the case of pure rotation all the singular values are equal to 1
        if (fabs(s1 - s3) < 1e-14) {
            res->push_back(std::make_tuple(Hs[i], V3D::Zero(), M3D()));
            continue;
        }
    
        // Compute orthogonal unit vectors
        V3D u1 = (sqrt(1.0 - s3) * v1 + sqrt(s1 - 1.0) * v3) / sqrt(s1 - s3);
        V3D u2 = (sqrt(1.0 - s3) * v1 - sqrt(s1 - 1.0) * v3) / sqrt(s1 - s3);

        M3D U1, W1, U2, W2;
        U1.col(0) = v2;
        U1.col(1) = u1;
        U1.col(2) = v2.cross(u1);

        W1.col(0) = H2 * v2;
        W1.col(1) = H2 * u1;
        W1.col(2) = (H2 * v2).cross(H2 * u1);

        U2.col(0) = v2;
        U2.col(1) = u2;
        U2.col(2) = v2.cross(u2);

        W2.col(0) = H2 * v2;
        W2.col(1) = H2 * u2;
        W2.col(2) = (H2 * v2).cross(H2 * u2);

        // compute the rotation matrices
        M3D R1 = W1 * U1.transpose();
        M3D R2 = W2 * U2.transpose();

        // build the solutions, discard those with negative plane normals
        // Compare to the original code, we do not invert the transformation.
        // Furthermore, we multiply t with -1.
        V3D n = v2.cross(u1);
        V3D t = -(H2 - R1) * n;
        res->push_back(std::make_tuple(R1, t, M3D()));

        t = (H2 - R1) * n;
        res->push_back(std::make_tuple(R1, t, M3D()));

        n = v2.cross(u2);
        t = -(H2 - R2) * n;
        res->push_back(std::make_tuple(R2, t, M3D()));

        t = (H2 - R2) * n;
        res->push_back(std::make_tuple(R2, t, M3D()));
    }
    return static_cast<int>(res->size());
}

inline double RelativePoseSolverHomography::ComputeOppositeOfMinor(
    const M3D& matrix_,
    const size_t row_,
    const size_t col_) const
{
    const size_t col1 = col_ == 0 ? 1 : 0;
    const size_t col2 = col_ == 2 ? 1 : 2;
    const size_t row1 = row_ == 0 ? 1 : 0;
    const size_t row2 = row_ == 2 ? 1 : 2;
    return (matrix_(row1, col2) * matrix_(row2, col1) - matrix_(row1, col1) * matrix_(row2, col2));
}

int RelativePoseSolverHomography::MinimalSolverWrapper(const std::vector<LineMatch>& line_matches,
                                                const std::vector<VPMatch>& vp_matches,
                                                const std::vector<JunctionMatch>& junction_matches,
                                                std::vector<ResultType>* res) const 
{
    std::vector<int> min_sample_sizes = min_sample_size();
    THROW_CHECK_EQ(line_matches.size(), min_sample_sizes[0]);
    THROW_CHECK_EQ(vp_matches.size(), min_sample_sizes[1]);
    THROW_CHECK_EQ(junction_matches.size(), min_sample_sizes[2]);

    std::vector<M3D> Hs;
    int num_H = HomographySolver(line_matches, vp_matches, junction_matches, &Hs);
    return DecomposeHomography(Hs, res);
}

}  // namespace line_relative_pose 

