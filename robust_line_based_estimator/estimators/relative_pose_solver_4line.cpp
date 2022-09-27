#include "estimators/relative_pose_solver_4line.h"

namespace line_relative_pose {

int RelativePoseSolver4line::HomographySolver(const std::vector<LineMatch>& line_matches,
                                              const std::vector<VPMatch>& vp_matches,
                                              const std::vector<PointMatch>& junction_matches,
                                              std::vector<M3D>* Hs) const 
{
    // TODO
    return 0;
}

}  // namespace line_relative_pose 

