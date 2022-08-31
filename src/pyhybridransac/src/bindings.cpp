#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "hybrid_ransac_python.h"

namespace py = pybind11;

PYBIND11_PLUGIN(pyhybridransac) {
    py::module m("pyhybridransac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pyhybridransac
        .. autosummary::
           :toctree: _generate

    )doc");

  return m.ptr();
}
