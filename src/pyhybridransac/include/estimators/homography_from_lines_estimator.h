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

#include "solver_homography_four_lines.h"

namespace hybridransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine, // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine, // The solver used for estimating the model from a non-minimal sample
			class _Sampler> // The sampler used for selecting a minimal sample 
		class HomographyFromLinesEstimator : public Estimator < cv::Mat, hybridransac::Model<3, 3> >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

			// The sampler used for selecting a minimal sample 
			_Sampler* const sampler;

			// The minimal sample used in sampling and then in the model estimation
			std::vector<std::pair<size_t, size_t>> minimal_sample;

			const bool use_endpoint_loss;
		public:
			HomographyFromLinesEstimator(
				_Sampler* const sampler_,
				std::vector<const cv::Mat*> containers_) :
				sampler(sampler_),
				minimal_solver(std::make_shared<_MinimalSolverEngine>(containers_)), // Minimal solver engine used for estimating a model from a minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>(containers_)), // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				use_endpoint_loss(false)
			{}

			~HomographyFromLinesEstimator() {}

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
				std::vector<Model>& models_)
			{
				// Select a minimal sample 
				sampler->sample(minimal_sample);

				// Estimate the models from the minimal sample
				minimal_solver->estimateModel(
					minimal_sample, // The sample used for the estimation
					models_); // The estimated model parameters

				return true;
			}

			// Estimating the model from a minimal sample
			inline bool estimateModel(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample usd for the estimation
				std::vector<Model>* models_) const // The estimated model parameters
			{
				return minimal_solver->estimateModel(data_, // The data points
					sample_, // The sample used for the estimation
					sampleSize(), // The size of a minimal sample
					*models_); // The estimated model parameters
			}

			inline double getLineParam(
				const double &vx_,
				const double &vy_,
				const double &x0_,
				const double &y0_,
				const double &x_,
				const double &y_) const
			{
				return (vy_ * y_ + vx_ * x_ - vy_ * y0_ - vx_ * x0_) / (vx_ * vx_ + vy_ * vy_);
			}			

			// Estimating the model from a non-minimal sample
			inline bool estimateModelNonminimal(const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				const size_t &sample_number_, // The size of a minimal sample
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const // The estimated model parameters
			{
				// Return of there are not enough points for the estimation
				if (sample_number_ < nonMinimalSampleSize())
					return false;
				
				return minimal_solver->estimateModel(data_, // The data points
					sample_, // The sample used for the estimation
					sample_number_, // The size of a minimal sample
					*models_); // The estimated model parameters
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
				const double r = residual(point_, descriptor_);
				return r * r;
			}

			static double residual(const cv::Mat& point_,
				const Model& model_)
			{
				return residual(point_, model_.descriptor);
			}

			static double residual(const cv::Mat& point_,
				const Eigen::Matrix3d& descriptor_)
			{
				const double* s = reinterpret_cast<double *>(point_.data);

				const double 
					&x1_src = *s,
					&y1_src = *(s + 1),
					&x2_src = *(s + 2),
					&y2_src = *(s + 3),
					&x1_dst = *(s + 4),
					&y1_dst = *(s + 5),
					&x2_dst = *(s + 6),
					&y2_dst = *(s + 7);

				/*if (use_endpoint_loss)
				{*/
					// Projecting the first end-point to the second image
					const double t31 = descriptor_(2, 0) * x1_src + descriptor_(2, 1) * y1_src + descriptor_(2, 2);
					const double t11 = (descriptor_(0, 0) * x1_src + descriptor_(0, 1) * y1_src + descriptor_(0, 2)) / t31;
					const double t21 = (descriptor_(1, 0) * x1_src + descriptor_(1, 1) * y1_src + descriptor_(1, 2)) / t31;
					 
					// Projecting the second end-point to the second image
					const double t32 = descriptor_(2, 0) * x2_src + descriptor_(2, 1) * y2_src + descriptor_(2, 2);
					const double t12 = (descriptor_(0, 0) * x2_src + descriptor_(0, 1) * y2_src + descriptor_(0, 2)) / t32;
					const double t22 = (descriptor_(1, 0) * x2_src + descriptor_(1, 1) * y2_src + descriptor_(1, 2)) / t32;

					const double dx11 = (t11 - x1_dst);
					const double dy11 = (t21 - y1_dst);
					const double d11 = sqrt(dx11 * dx11 + dy11 * dy11); 
					
					const double dx21 = (t11 - x2_dst);
					const double dy21 = (t21 - y2_dst);
					const double d21 = sqrt(dx21 * dx21 + dy21 * dy21); 

					const double d1 = MIN(d11, d21);

					const double dx12 = (t12 - x1_dst);
					const double dy12 = (t22 - y1_dst);
					const double d12 = sqrt(dx12 * dx12 + dy12 * dy12); 
					
					const double dx22 = (t12 - x2_dst);
					const double dy22 = (t22 - y2_dst);
					const double d22 = sqrt(dx22 * dx22 + dy22 * dy22); 

					const double d2 = MIN(d12, d22);

					return 0.5 * (d1 + d2);

				/* } else
				{
					// Calculating the line eq in the second image
					double vx2 = x2_dst - x1_dst,
						vy2 = y2_dst - y1_dst;
					const double len2 = sqrt(vx2 * vx2 + vy2 * vy2);
					//vx2 /= len2;
					//vy2 /= len2;
					const double a2 = -vy2,
						b2 = vx2;
					const double c2 = -a2 * x1_dst - b2 * y1_dst;

					const double t31 = descriptor_(2, 0) * x1_src + descriptor_(2, 1) * y1_src + descriptor_(2, 2);
					const double t11 = (descriptor_(0, 0) * x1_src + descriptor_(0, 1) * y1_src + descriptor_(0, 2)) / t31;
					const double t21 = (descriptor_(1, 0) * x1_src + descriptor_(1, 1) * y1_src + descriptor_(1, 2)) / t31;
					
					double t1 = getLineParam(vx2, vy2, x1_dst, y1_dst, t11, t21);
					double d1;
					d1 = abs(t11 * a2 + t21 * b2 + c2) / len2;

					// Projecting the second end-point to the second image
					const double t32 = descriptor_(2, 0) * x2_src + descriptor_(2, 1) * y2_src + descriptor_(2, 2);
					const double t12 = (descriptor_(0, 0) * x2_src + descriptor_(0, 1) * y2_src + descriptor_(0, 2)) / t32;
					const double t22 = (descriptor_(1, 0) * x2_src + descriptor_(1, 1) * y2_src + descriptor_(1, 2)) / t32;
					
					double t2 = getLineParam(vx2, vy2, x1_dst, y1_dst, t12, t22);
					double d2;
					d2 = abs(t12 * a2 + t22 * b2 + c2) / len2;

					return 0.5 * (d1 + d2);
				}*/
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
					model.descriptor.block<3, 3>(0, 0).determinant();

				// Check if the homography has a small determinant.
				constexpr double kMinimumDeterminant = 1e-10;
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
				return true;
			}
		};
	}
}