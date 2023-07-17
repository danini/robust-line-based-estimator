set(FOLDER_NAME "refinement")

LINE_RELATIVE_POSE_ADD_SOURCES(
    cost_functions.h
    ls_sampson.h ls_sampson.cpp
    ls_combined.h ls_combined.cpp
    vp_refinement.h vp_refinement.cpp
    ls_sampson_plus_association.h ls_sampson_plus_association.cpp
)

