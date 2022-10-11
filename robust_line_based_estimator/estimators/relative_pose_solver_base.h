#ifndef LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_BASE_H_ 
#define LINE_RELATIVE_POSE_ESTIMATORS_RELATIVE_POSE_SOLVER_BASE_H_ 

#include <Eigen/Core>
#include <Eigen/Dense>

#include "base/types.h"
#include "base/junction.h"

namespace line_relative_pose {

// Only supporting calibrated cases
// The input lines, vps and junctions are already normalized
class RelativePoseSolverBase {
public:
    RelativePoseSolverBase() {}
    ~RelativePoseSolverBase() {}
    using ResultType = std::pair<M3D, V3D>;

    virtual std::vector<int> min_sample_size() const = 0;

    virtual int MinimalSolverWrapper(const std::vector<LineMatch>& line_matches, 
                                     const std::vector<VPMatch>& vp_matches,
                                     const std::vector<JunctionMatch>& junction_matches,
                                     std::vector<ResultType>* res) const = 0;

};

}  // namespace line_relative_pose 

#endif 

