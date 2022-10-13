#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_HOMOGRAPHY_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_HOMOGRAPHY_H_ 

#include <Eigen/Core>

#include "estimators/relative_pose_solver_base.h"

namespace line_relative_pose {

// homography-based essential matrix
class RelativePoseSolverHomography: public RelativePoseSolverBase {
public:
    RelativePoseSolverHomography(): RelativePoseSolverBase() {}
    using ResultType = RelativePoseSolverBase::ResultType;

    int MinimalSolverWrapper(const std::vector<LineMatch>& line_matches, 
                             const std::vector<VPMatch>& vp_matches,
                             const std::vector<JunctionMatch>& junction_matches,
                             std::vector<ResultType>* res) const override;

    // return a set of homography from image 1 to image 2.
    virtual int HomographySolver(const std::vector<LineMatch>& line_matches, 
                                 const std::vector<VPMatch>& vp_matches,
                                 const std::vector<JunctionMatch>& junction_matches,
                                 std::vector<M3D>* Hs) const = 0;

protected:
    int DecomposeHomography(const std::vector<M3D>& Hs, std::vector<ResultType>* res) const;

    inline double ComputeOppositeOfMinor(
        const Eigen::Matrix3d& matrix_,
        const size_t row_,
        const size_t col_) const;
};

}  // namespace line_relative_pose 

#endif 

