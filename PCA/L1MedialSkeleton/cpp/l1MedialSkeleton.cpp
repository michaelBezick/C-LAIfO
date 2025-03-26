#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <immintrin.h>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

void process_array_cuda(double *h_arr, int size);

namespace py = pybind11;
using namespace Eigen;
using namespace std;

double theta(double r, double h) {
  return exp(-(r * r) / ((h / 2.0) * (h / 2.0)));
}

py::array_t<double> computeCovarianceMatrix(py::array_t<double> points,
                                            float h) {

  auto buf = points.request();

  if (buf.ndim != 2 || buf.shape[1] != 3) {
    throw std::runtime_error("Input array must be of shape (N,3)");
  }

  int I = buf.shape[0];
  auto ptr = static_cast<double *>(buf.ptr);

  // converting input into Eigen 3D points
  vector<Vector3d> pointVec(I);
  for (int i = 0; i < I; i++) {
    pointVec[i] = Vector3d(ptr[3 * i], ptr[3 * i + 1], ptr[3 * i + 2]);
  }

  vector<Matrix3d> C(I, Matrix3d::Zero());
#pragma omp parallel for
  for (int i = 0; i < I; i++) {
    Matrix3d sum = Matrix3d::Zero();

    for (int j = 0; j < I; j++) {
      if (i == j)
        continue;

      Vector3d diff = pointVec[i] - pointVec[j];
      double dist = diff.norm();
      double weight = theta(dist, h);

      sum += weight * (diff * diff.transpose());
    }
    C[i] = sum;
  }

  // convert back to numpy array
  py::array_t<double> result({I, 3, 3});
  auto rbuf = result.request();
  double *res_ptr = static_cast<double *>(rbuf.ptr);

  for (int i = 0; i < I; i++) {
    memcpy(res_ptr + (i * 9), C[i].data(), 9 * sizeof(double));
  }

  return result;
}

// Bind functions to Python module
PYBIND11_MODULE(l1MedialSkeleton, m) {
  m.def("computeCovarianceMatrix", &computeCovarianceMatrix,
        "Compute covariance matrix for a point cloud", py::arg("points"),
        py::arg("h"));
}
