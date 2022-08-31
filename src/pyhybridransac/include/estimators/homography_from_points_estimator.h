// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

#include "solver_homography_four_point.h"

namespace hybridransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine, // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine, // The solver used for estimating the model from a non-minimal sample
			class _Sampler> // The sampler used for selecting a minimal sample 
			class HomographyFromPointsEstimator : public Estimator < cv::Mat, hybridransac::Model<3, 3> >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

			// The sampler used for selecting a minimal sample 
			_Sampler * const sampler;

			// The minimal sample used in sampling and then in the model estimation
			std::vector<std::pair<size_t, size_t>> minimal_sample;

		public:
			HomographyFromPointsEstimator(
				_Sampler * const sampler_,
				std::vector<const cv::Mat *> containers_) :
				sampler(sampler_),
				minimal_solver(std::make_shared<_MinimalSolverEngine>(containers_)), // Minimal solver engine used for estimating a model from a minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>(containers_)) // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			{
			}

			~HomographyFromPointsEstimator() 
			{
			}

			// Return the minimal solver
			const _MinimalSolverEngine *getMinimalSolver() const
			{
				return minimal_solver.get();
			}

			// Return a mutable minimal solver
			_MinimalSolverEngine *getMutableMinimalSolver()
			{
				return minimal_solver.get();
			}

			// Return the minimal solver
			const _NonMinimalSolverEngine *getNonMinimalSolver() const
			{
				return non_minimal_solver.get();
			}

			// Return a mutable minimal solver
			_NonMinimalSolverEngine *getMutableNonMinimalSolver()
			{
				return non_minimal_solver.get();
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			inline size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Running both the sampling and the model estimation
			inline bool runMinimalSolver(
				std::vector<Model> &models_)
			{
				// Select a minimal sample 
				sampler->sample(minimal_sample);

				// Estimate the models from the minimal sample
				minimal_solver->estimateModel(
					minimal_sample, // The sample used for the estimation
					models_); // The estimated model parameters

				return true;
			}

			static double squaredResidual(const cv::Mat& point_,
				const Model& model_)
			{
				return squaredResidual(point_, model_.descriptor);
			}

			static double squaredResidual(const cv::Mat& point_,
				const Eigen::Matrix3d& descriptor_)
			{
				const double* s = reinterpret_cast<double *>(point_.data);

				const double &x1 = *s,
					&y1 = *(s + 1),
					&x2 = *(s + 2),
					&y2 = *(s + 3);

				const double t1 = descriptor_(0, 0) * x1 + descriptor_(0, 1) * y1 + descriptor_(0, 2);
				const double t2 = descriptor_(1, 0) * x1 + descriptor_(1, 1) * y1 + descriptor_(1, 2);
				const double t3 = descriptor_(2, 0) * x1 + descriptor_(2, 1) * y1 + descriptor_(2, 2);

				const double d1 = x2 - (t1 / t3);
				const double d2 = y2 - (t2 / t3);

				return d1 * d1 + d2 * d2;
			}

			static double residual(const cv::Mat& point_,
				const hybridransac::Model<3, 3>& model_)
			{
				return residual(point_, model_.descriptor);
			}

			static double residual(const cv::Mat& point_,
				const Eigen::Matrix3d& descriptor_)
			{
				return sqrt(squaredResidual(point_, descriptor_));
			}

			inline bool normalizePoints(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The points to which the model will be fit
				const size_t &sample_number_,// The number of points
				cv::Mat &normalized_points_, // The normalized point coordinates
				Eigen::Matrix3d &normalizing_transform_source_, // The normalizing transformation in the first image
				Eigen::Matrix3d &normalizing_transform_destination_) const // The normalizing transformation in the second image
			{
				const size_t cols = data_.cols;
				double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
				const double *points_ptr = reinterpret_cast<double *>(data_.data);

				double mass_point_src[2], // Mass point in the first image
					mass_point_dst[2]; // Mass point in the second image

				// Initializing the mass point coordinates
				mass_point_src[0] =
					mass_point_src[1] =
					mass_point_dst[0] =
					mass_point_dst[1] =
					0.0;

				// Calculating the mass points in both images
				for (size_t i = 0; i < sample_number_; ++i)
				{
					// Get pointer of the current point
					const double *d_idx = points_ptr + cols * sample_[i];

					// Add the coordinates to that of the mass points
					mass_point_src[0] += *(d_idx);
					mass_point_src[1] += *(d_idx + 1);
					mass_point_dst[0] += *(d_idx + 2);
					mass_point_dst[1] += *(d_idx + 3);
				}

				// Get the average
				mass_point_src[0] /= sample_number_;
				mass_point_src[1] /= sample_number_;
				mass_point_dst[0] /= sample_number_;
				mass_point_dst[1] /= sample_number_;

				// Get the mean distance from the mass points
				double average_distance_src = 0.0,
					average_distance_dst = 0.0;
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const double *d_idx = points_ptr + cols * sample_[i];

					const double &x1 = *(d_idx);
					const double &y1 = *(d_idx + 1);
					const double &x2 = *(d_idx + 2);
					const double &y2 = *(d_idx + 3);

					const double dx1 = mass_point_src[0] - x1;
					const double dy1 = mass_point_src[1] - y1;
					const double dx2 = mass_point_dst[0] - x2;
					const double dy2 = mass_point_dst[1] - y2;

					average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
					average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
				}

				average_distance_src /= sample_number_;
				average_distance_dst /= sample_number_;

				// Calculate the sqrt(2) / MeanDistance ratios
				const double ratio_src = M_SQRT2 / average_distance_src;
				const double ratio_dst = M_SQRT2 / average_distance_dst;

				// Compute the normalized coordinates
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const double *d_idx = points_ptr + cols * sample_[i];

					const double &x1 = *(d_idx);
					const double &y1 = *(d_idx + 1);
					const double &x2 = *(d_idx + 2);
					const double &y2 = *(d_idx + 3);

					*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
					*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
					*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
					*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;

					for (size_t i = 4; i < normalized_points_.cols; ++i)
						*normalized_points_ptr++ = *(d_idx + i);
				}

				// Creating the normalizing transformations
				normalizing_transform_source_ << ratio_src, 0, -ratio_src * mass_point_src[0],
					0, ratio_src, -ratio_src * mass_point_src[1],
					0, 0, 1;

				normalizing_transform_destination_ << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
					0, ratio_dst, -ratio_dst * mass_point_dst[1],
					0, 0, 1;
				return true;
			}

			// Calculates the cross-product of two vectors
			inline void cross_product(
				Eigen::Vector3d &result_,
				const double *vector1_,
				const double *vector2_,
				const unsigned int st_) const
			{
				result_[0] = vector1_[st_] - vector2_[st_];
				result_[1] = vector2_[0] - vector1_[0];
				result_[2] = vector1_[0] * vector2_[st_] - vector1_[st_] * vector2_[0];
			}

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			inline bool isValidModel(Model& model,
				const Datum& data,
				const std::vector<size_t>& inliers,
				const size_t* minimal_sample_,
				const double threshold_,
				bool& model_updated_) const
			{ 
				// Calculate the determinant of the homography
				const double kDeterminant =
					model.descriptor.determinant();

				// Check if the homography has a small determinant.
				constexpr double kMinimumDeterminant = 1e-2;
				if (abs(kDeterminant) < kMinimumDeterminant)
					return false;
				return true;
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			inline bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t *sample_) const // The indices of the selected points
			{
				// The size of a minimal sample
				constexpr size_t sample_size = sampleSize();

				// Check oriented constraints
				Eigen::Vector3d p, q;

				const double *a = reinterpret_cast<const double *>(data_.row(sample_[0]).data),
					*b = reinterpret_cast<const double *>(data_.row(sample_[1]).data),
					*c = reinterpret_cast<const double *>(data_.row(sample_[2]).data),
					*d = reinterpret_cast<const double *>(data_.row(sample_[3]).data);

				cross_product(p, a, b, 1);
				cross_product(q, a + 2, b + 2, 1);

				if ((p[0] * c[0] + p[1] * c[1] + p[2])*(q[0] * c[2] + q[1] * c[3] + q[2]) < 0)
					return false;
				if ((p[0] * d[0] + p[1] * d[1] + p[2])*(q[0] * d[2] + q[1] * d[3] + q[2]) < 0)
					return false;

				cross_product(p, c, d, 1);
				cross_product(q, c + 2, d + 2, 1);

				if ((p[0] * a[0] + p[1] * a[1] + p[2])*(q[0] * a[2] + q[1] * a[3] + q[2]) < 0)
					return false;
				if ((p[0] * b[0] + p[1] * b[1] + p[2])*(q[0] * b[2] + q[1] * b[3] + q[2]) < 0)
					return false;

				return true;
			}
		};
	}
}