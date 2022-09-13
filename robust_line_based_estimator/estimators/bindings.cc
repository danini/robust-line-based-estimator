#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <RansacLib/sampling.h>
#include <RansacLib/ransac.h>
#include <Eigen/Core>

void bind_estimators(py::module& m) {}

void bind_ransaclib(py::module& m) {
    py::class_<ransac_lib::RansacStatistics>(m, "RansacStats")
        .def(py::init<>())
        .def_readwrite("num_iterations", &ransac_lib::RansacStatistics::num_iterations)
        .def_readwrite("best_num_inliers", &ransac_lib::RansacStatistics::best_num_inliers)
        .def_readwrite("best_model_score", &ransac_lib::RansacStatistics::best_model_score)
        .def_readwrite("inlier_ratio", &ransac_lib::RansacStatistics::inlier_ratio)
        .def_readwrite("inlier_indices", &ransac_lib::RansacStatistics::inlier_indices)
        .def_readwrite("number_lo_iterations", &ransac_lib::RansacStatistics::number_lo_iterations);
    
    py::class_<ransac_lib::RansacOptions>(m, "RansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ransac_lib::RansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ransac_lib::RansacOptions::max_num_iterations_)
        .def_readwrite("success_probability_", &ransac_lib::RansacOptions::success_probability_)
        .def_readwrite("squared_inlier_threshold_", &ransac_lib::RansacOptions::squared_inlier_threshold_)
        .def_readwrite("random_seed_", &ransac_lib::RansacOptions::random_seed_);

    py::class_<ransac_lib::LORansacOptions>(m, "LORansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ransac_lib::LORansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ransac_lib::LORansacOptions::max_num_iterations_)
        .def_readwrite("success_probability_", &ransac_lib::LORansacOptions::success_probability_)
        .def_readwrite("squared_inlier_threshold_", &ransac_lib::LORansacOptions::squared_inlier_threshold_)
        .def_readwrite("random_seed_", &ransac_lib::LORansacOptions::random_seed_)
        .def_readwrite("num_lo_steps_", &ransac_lib::LORansacOptions::num_lo_steps_)
        .def_readwrite("threshold_multiplier_", &ransac_lib::LORansacOptions::threshold_multiplier_)
        .def_readwrite("num_lsq_iterations_", &ransac_lib::LORansacOptions::num_lsq_iterations_)
        .def_readwrite("min_sample_multiplicator_", &ransac_lib::LORansacOptions::min_sample_multiplicator_)
        .def_readwrite("non_min_sample_multiplier_", &ransac_lib::LORansacOptions::non_min_sample_multiplier_)
        .def_readwrite("lo_starting_iterations_", &ransac_lib::LORansacOptions::lo_starting_iterations_)
        .def_readwrite("final_least_squares_", &ransac_lib::LORansacOptions::final_least_squares_);
}

PYBIND11_MODULE(uncalibrated_vp_estimators, m){
    m.doc() = "pybind11 for line-based relative pose estimation";
    bind_ransaclib(m);
    bind_estimators(m);
}

