set(FOLDER_NAME "estimators")

LINE_RELATIVE_POSE_ADD_SOURCES(
    bindings.cc
    functions.h functions.cpp
    line_relative_pose_hybrid_ransac.h
    hybrid_relative_pose_estimator_base.h hybrid_relative_pose_estimator_base.cpp
    hybrid_relative_pose_estimator.h hybrid_relative_pose_estimator.cpp
    relative_pose_solver_base.h
    relative_pose_solver_5pt.h relative_pose_solver_5pt.cpp
    relative_pose_solver_homography.h relative_pose_solver_homography.cpp
    relative_pose_solver_4line.h relative_pose_solver_4line.cpp
    relative_pose_solver_1vp_3pt.h relative_pose_solver_1vp_3pt.cpp
    relative_pose_solver_1vp_3cll.h relative_pose_solver_1vp_3cll.cpp
    relative_pose_solver_2vp_2pt.h relative_pose_solver_2vp_2pt.cpp
    relative_pose_solver_2vp_3cll.h relative_pose_solver_2vp_3cll.cpp

    relative_pose_solver_1line_1vp_2pt_orthogonal.h relative_pose_solver_1line_1vp_2pt_orthogonal.cpp
    relative_pose_solver_1vp_2pt_orthogonal.h relative_pose_solver_1vp_2pt_orthogonal.cpp
    relative_pose_solver_1vp_3cll_orthogonal.h relative_pose_solver_1vp_3cll_orthogonal.cpp
)

