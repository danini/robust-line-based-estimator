set(FOLDER_NAME "solvers")

LINE_RELATIVE_POSE_ADD_SOURCES(
    sturm.h charpoly.h
    essential_matrix_coeffs.h essential_matrix_poly.h
    5pt_solver.hxx
    solver_H.h

    solver_wrapper_1vp_222cl.h
    solver_1vp_222cl.cpp
    helper_functions_2vp_3cl.h
    solver_1vp_3pt.h

    solver_wrapper_2vp_22cl.h
    solver_2vp_22cl.cpp
    solver_2vp_2pt.h
    solver_2vp_cl_3stage.h
    solver_cpp_2vp_3cl_hidden.cpp
)

