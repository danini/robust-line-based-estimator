#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <chrono>

//#include <ransac.hxx>
//#include <homotopy.hxx>
//#include "essential_matrix_poly.h"
//#include "essential_matrix_coeffs.h"

//PROTOTYPES
//the 5pt solver
//returns number of real solutions
int essential_solver(const Eigen::Vector2d points1[5], //array of 5 points in the 1st camera
					 const Eigen::Vector2d points2[5], //array of 5 points in the 2nd camera
					 Eigen::Matrix3d * Es); //output array of at most 10 essential matrices consistent with the input
					 
//decomposes the essential matrix into the rotation and translation
//returns number of valid poses
int decompose_essential(const Eigen::Matrix3d E, //essential matrix
						Eigen::Matrix3d * Rs, //array of at most 4 output rotation matrices
						Eigen::Vector3d * ts); //array of at most 4 output translation vectors

bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs_all,
                                    Eigen::VectorXd* real,
                                    Eigen::VectorXd* imag);

//FUNCTIONS
int essential_solver(const Eigen::Vector2d points1[5], const Eigen::Vector2d points2[5], Eigen::Matrix3d * Es)
{
	//obtain homogeneous points
	Eigen::Vector3d P1[5];
	Eigen::Vector3d Q1[5];
	for(int i=0;i<5;++i)
	{
		P1[i] = points1[i].homogeneous();
		Q1[i] = points2[i].homogeneous();
	}

	//solver

	// Step 1: Extraction of the nullspace x, y, z, w.

	Eigen::Matrix<double, Eigen::Dynamic, 9> Q(5, 9);
	for (size_t i = 0; i < 5; ++i)
	{
		const double x1_0 = points1[i](0);
		const double x1_1 = points1[i](1);
		const double x2_0 = points2[i](0);
		const double x2_1 = points2[i](1);
		Q(i, 0) = x1_0 * x2_0;
		Q(i, 1) = x1_1 * x2_0;
		Q(i, 2) = x2_0;
		Q(i, 3) = x1_0 * x2_1;
		Q(i, 4) = x1_1 * x2_1;
		Q(i, 5) = x2_1;
		Q(i, 6) = x1_0;
		Q(i, 7) = x1_1;
		Q(i, 8) = 1;
	}

	// Extract the 4 Eigen vectors corresponding to the smallest singular values.
	const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(Q, Eigen::ComputeFullV);
	const Eigen::Matrix<double, 9, 4> E = svd.matrixV().block<9, 4>(0, 5);

	// Step 3: Gauss-Jordan elimination with partial pivoting on A.

	Eigen::Matrix<double, 10, 20> A;
	#include "essential_matrix_poly.h"
	Eigen::Matrix<double, 10, 10> AA =
	  A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

	// Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
	//         matrix B to obtain the tenth degree polynomial.

	Eigen::Matrix<double, 13, 3> B;
	for (size_t i = 0; i < 3; ++i)
	{
		B(0, i) = 0;
		B(4, i) = 0;
		B(8, i) = 0;
		B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
		B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
		B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
		B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
		B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
		B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
	}

	// Step 5: Extraction of roots from the degree 10 polynomial.
	Eigen::Matrix<double, 11, 1> coeffs;
	#include "essential_matrix_coeffs.h"

	Eigen::VectorXd roots_real;
	Eigen::VectorXd roots_imag;
	if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag))
	{
		//return {};
		return 0;
	}

	std::vector<Eigen::Matrix3d> models;
	models.reserve(roots_real.size());

	int pos = 0;
	for (Eigen::VectorXd::Index i = 0; i < roots_imag.size(); ++i)
	{
		const double kMaxRootImag = 1e-10;
		if (std::abs(roots_imag(i)) > kMaxRootImag)
		{
		  continue;
		}

		const double z1 = roots_real(i);
		const double z2 = z1 * z1;
		const double z3 = z2 * z1;
		const double z4 = z3 * z1;

		Eigen::Matrix3d Bz;
		for (size_t j = 0; j < 3; ++j) 
		{
			Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
			Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
			Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 + B(12, j);
		}

		const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
		const Eigen::Vector3d X = svd.matrixV().block<3, 1>(0, 2);

		const double kMaxX3 = 1e-10;
		if (std::abs(X(2)) < kMaxX3)
		{
			continue;
		}

		Eigen::MatrixXd essential_vec = E.col(0) * (X(0) / X(2)) +
							            E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
							            E.col(3);
		essential_vec /= essential_vec.norm();

		const Eigen::Matrix3d essential_matrix = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(essential_vec.data());
		Es[pos] = essential_matrix;
		++pos;
		//models.push_back(essential_matrix);
	}

	//return models;
	return pos;
}

// Remove leading zero coefficients.
Eigen::VectorXd RemoveLeadingZeros(const Eigen::VectorXd& coeffs)
{
	Eigen::VectorXd::Index num_zeros = 0;
	for (; num_zeros < coeffs.size(); ++num_zeros)
	{
		if (coeffs(num_zeros) != 0)
		{
			break;
		}
	}
	return coeffs.tail(coeffs.size() - num_zeros);
}

// Remove trailing zero coefficients.
Eigen::VectorXd RemoveTrailingZeros(const Eigen::VectorXd& coeffs)
{
	Eigen::VectorXd::Index num_zeros = 0;
	for (; num_zeros < coeffs.size(); ++num_zeros)
	{
		if (coeffs(coeffs.size() - 1 - num_zeros) != 0)
		{
			break;
		}
	}
	return coeffs.head(coeffs.size() - num_zeros);
}

bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs, Eigen::VectorXd* real, Eigen::VectorXd* imag)
{
	//CHECK_EQ(coeffs.size(), 2);

	if (coeffs(0) == 0)
	{
		return false;
	}

	if (real != nullptr)
	{
		real->resize(1);
		(*real)(0) = -coeffs(1) / coeffs(0);
	}

	if (imag != nullptr)
	{
		imag->resize(1);
		(*imag)(0) = 0;
	}

	return true;
}

bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs, Eigen::VectorXd* real, Eigen::VectorXd* imag)
{
  //CHECK_EQ(coeffs.size(), 3);

  const double a = coeffs(0);
  if (a == 0) {
	return FindLinearPolynomialRoots(coeffs.tail(2), real, imag);
  }

  const double b = coeffs(1);
  const double c = coeffs(2);
  if (b == 0 && c == 0) {
	if (real != nullptr) {
	  real->resize(1);
	  (*real)(0) = 0;
	}
	if (imag != nullptr) {
	  imag->resize(1);
	  (*imag)(0) = 0;
	}
	return true;
  }

  const double d = b * b - 4 * a * c;

  if (d >= 0) {
	const double sqrt_d = std::sqrt(d);
	if (real != nullptr) {
	  real->resize(2);
	  if (b >= 0) {
	    (*real)(0) = (-b - sqrt_d) / (2 * a);
	    (*real)(1) = (2 * c) / (-b - sqrt_d);
	  } else {
	    (*real)(0) = (2 * c) / (-b + sqrt_d);
	    (*real)(1) = (-b + sqrt_d) / (2 * a);
	  }
	}
	if (imag != nullptr) {
	  imag->resize(2);
	  imag->setZero();
	}
  } else {
	if (real != nullptr) {
	  real->resize(2);
	  real->setConstant(-b / (2 * a));
	}
	if (imag != nullptr) {
	  imag->resize(2);
	  (*imag)(0) = std::sqrt(-d) / (2 * a);
	  (*imag)(1) = -(*imag)(0);
	}
  }

  return true;
}

/*bool FindPolynomialRootsDurandKerner(const Eigen::VectorXd& coeffs_all, Eigen::VectorXd* real, Eigen::VectorXd* imag) const
{
  //CHECK_GE(coeffs_all.size(), 2);

  const Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);

  const int degree = coeffs.size() - 1;

  if (degree <= 0) {
	return false;
  } else if (degree == 1) {
	return FindLinearPolynomialRoots(coeffs, real, imag);
  } else if (degree == 2) {
	return FindQuadraticPolynomialRoots(coeffs, real, imag);
  }

  // Initialize roots.
  Eigen::VectorXcd roots(degree);
  roots(degree - 1) = std::complex<double>(1, 0);
  for (int i = degree - 2; i >= 0; --i) {
	roots(i) = roots(i + 1) * std::complex<double>(1, 1);
  }

  // Iterative solver.
  const int kMaxNumIterations = 100;
  const double kMaxRootChange = 1e-10;
  for (int iter = 0; iter < kMaxNumIterations; ++iter) {
	double max_root_change = 0.0;
	for (int i = 0; i < degree; ++i) {
	  const std::complex<double> root_i = roots(i);
	  std::complex<double> numerator = coeffs[0];
	  std::complex<double> denominator = coeffs[0];
	  for (int j = 0; j < degree; ++j) {
	    numerator = numerator * root_i + coeffs[j + 1];
	    if (i != j) {
	      denominator = denominator * (root_i - roots(j));
	    }
	  }
	  const std::complex<double> root_i_change = numerator / denominator;
	  roots(i) = root_i - root_i_change;
	  max_root_change =
	      std::max(max_root_change, std::abs(root_i_change.real()));
	  max_root_change =
	      std::max(max_root_change, std::abs(root_i_change.imag()));
	}

	// Break, if roots do not change anymore.
	if (max_root_change < kMaxRootChange) {
	  break;
	}
  }

  if (real != nullptr) {
	real->resize(degree);
	*real = roots.real();
  }
  if (imag != nullptr) {
	imag->resize(degree);
	*imag = roots.imag();
  }

  return true;
}*/

bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs_all,
                                    Eigen::VectorXd* real,
                                    Eigen::VectorXd* imag)
{

	Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);

	const int degree = coeffs.size() - 1;

	if (degree <= 0)
	{
		return false;
	}
	else if (degree == 1)
	{
		return FindLinearPolynomialRoots(coeffs, real, imag);
	}
	else if (degree == 2)
	{
		return FindQuadraticPolynomialRoots(coeffs, real, imag);
	}

	// Remove the coefficients where zero is a solution.
	coeffs = RemoveTrailingZeros(coeffs);

	// Check if only zero is a solution.
	if (coeffs.size() == 1)
	{
		if (real != nullptr)
		{
			real->resize(1);
	  		(*real)(0) = 0;
		}
		if (imag != nullptr)
		{
			imag->resize(1);
			(*imag)(0) = 0;
		}
		return true;
	}

	// Fill the companion matrix.
	Eigen::MatrixXd C(coeffs.size() - 1, coeffs.size() - 1);
	C.setZero();
	for (Eigen::MatrixXd::Index i = 1; i < C.rows(); ++i)
	{
		C(i, i - 1) = 1;
	}
	C.row(0) = -coeffs.tail(coeffs.size() - 1) / coeffs(0);

	// Solve for the roots of the polynomial.
	Eigen::EigenSolver<Eigen::MatrixXd> solver(C, false);
	if (solver.info() != Eigen::Success)
	{
		return false;
	}

	// If there are trailing zeros, we must add zero as a solution.
	const int effective_degree = coeffs.size() - 1 < degree ? coeffs.size() : coeffs.size() - 1;

	if (real != nullptr)
	{
		real->resize(effective_degree);
		real->head(coeffs.size() - 1) = solver.eigenvalues().real();
		if (effective_degree > coeffs.size() - 1)
		{
			(*real)(real->size() - 1) = 0;
		}
	}
	if (imag != nullptr)
	{
		imag->resize(effective_degree);
		imag->head(coeffs.size() - 1) = solver.eigenvalues().imag();
		if (effective_degree > coeffs.size() - 1)
		{
			(*imag)(imag->size() - 1) = 0;
		}
	}

	return true;
}

int decompose_essential(const Eigen::Matrix3d E, Eigen::Matrix3d * Rs, Eigen::Vector3d * ts)
{
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Matrix3d W1;
	W1 << 0,1,0,-1,0,0,0,0,1;
	Eigen::Matrix3d W2;
	W2 << 0,-1,0,1,0,0,0,0,1;
	Eigen::Matrix3d W3;
	W3 << 0,1,0,-1,0,0,0,0,-1;
	Eigen::Matrix3d W4;
	W4 << 0,-1,0,1,0,0,0,0,-1;
	Eigen::Matrix3d R1 = U*W1*V.transpose();
	Eigen::Matrix3d R2 = U*W2*V.transpose();
	double det = R1.determinant();
	if(det < 0)
	{
		R1 = U*W3*V.transpose();
		R2 = U*W4*V.transpose();
	}
	Eigen::Vector3d t = U.col(2);
	
	Rs[0] = R1;
	ts[0] = t;
	Rs[1] = R2;
	ts[1] = t;
	
	Rs[2] = R1;
	ts[2] = -t;
	Rs[3] = R2;
	ts[3] = -t;
	
	return 4;
}

