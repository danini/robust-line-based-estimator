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

#include "solver_engine.h"

namespace hybridransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class HomographyFourLineSolver : public SolverEngine<3, 3>
			{
			protected:
				std::vector<const cv::Mat*> dataContainers;

			public:
				HomographyFourLineSolver(std::vector<const cv::Mat*> dataContainers_) :
					dataContainers(dataContainers_)
				{
				}

				~HomographyFourLineSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel<3, 3>' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}
				
				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 4;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				inline bool estimateModel(
					const std::vector<std::pair<size_t, size_t>>& sample_,
					std::vector<Model<3, 3>> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				inline bool estimateNonMinimalModel(
					const std::vector<std::pair<size_t, size_t>>& sample_,
					std::vector<Model<3, 3>>& models_,
					const double* weights_) const;

				inline bool estimateMinimalModel(
					const std::vector<std::pair<size_t, size_t>>& sample_,
					std::vector<Model<3, 3>>& models_,
					const double* weights_) const;

				inline void getImplicitLine(
					const double &x1,
					const double &y1,
					const double &x2,
					const double &y2,
					double &a,
					double &b,
					double &c) const;
			};

			inline void HomographyFourLineSolver::getImplicitLine(
				const double &x1,
				const double &y1,
				const double &x2,
				const double &y2,
				double &a,
				double &b,
				double &c) const
			{
				// Calculating the line eq
				double vx = x2 - x1,
					vy = y2 - y1;
				a = -vy;
				b = vx;
				c = -a * x1 - b * y1;			
			}

			inline bool HomographyFourLineSolver::estimateMinimalModel(
				const std::vector<std::pair<size_t, size_t>>& sample_,
				std::vector<Model<3, 3>>& models_,
				const double* weights_) const
			{
				constexpr size_t equation_number = 2;
				Eigen::Matrix<double, 8, 9> coefficients;

				const size_t& kContainerIdx = sample_[0].first;
				const auto& kDataContainer = dataContainers[kContainerIdx];
				const size_t columns = kDataContainer->cols;
				const double* data_ptr = reinterpret_cast<double*>(kDataContainer->data);
				size_t row_idx = 0;
				double weight = 1.0;

				for (size_t i = 0; i < sample_.size(); ++i)
				{
					const size_t &idx = sample_[i].second;

					const double* point_ptr =
						data_ptr + idx * columns;

					const double
						& x11 = point_ptr[0],
						& y11 = point_ptr[1],
						& x12 = point_ptr[2],
						& y12 = point_ptr[3],
						& x21 = point_ptr[4],
						& y21 = point_ptr[5],
						& x22 = point_ptr[6],
						& y22 = point_ptr[7];

					if (weights_ != nullptr)
						weight = weights_[idx];

					double a1, b1, c1,
						a2, b2, c2;

					getImplicitLine(x11, y11, x12, y12, a1, b1, c1);
					getImplicitLine(x21, y21, x22, y22, a2, b2, c2);

					a1 /= c1;
					b1 /= c1;
					a2 /= c2;
					b2 /= c2;
					
					// TODO(danini): Add weighting
					coefficients(row_idx, 0) = -a1;
					coefficients(row_idx, 1) = 0;
					coefficients(row_idx, 2) = a1 * a2;
					coefficients(row_idx, 3) = -b1;
					coefficients(row_idx, 4) = 0;
					coefficients(row_idx, 5) = b1 * a2;
					coefficients(row_idx, 6) = -1;
					coefficients(row_idx, 7) = 0;
					coefficients(row_idx, 8) = a2;
					++row_idx;

					coefficients(row_idx, 0) = 0;
					coefficients(row_idx, 1) = -a1;
					coefficients(row_idx, 2) = a1 * b2;
					coefficients(row_idx, 3) = 0;
					coefficients(row_idx, 4) = -b1;
					coefficients(row_idx, 5) = -b1 * b2;
					coefficients(row_idx, 6) = 0;
					coefficients(row_idx, 7) = -1;
					coefficients(row_idx, 8) = b2;
					++row_idx;
				}
								
				const Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(
					coefficients.transpose() * coefficients);
				const Eigen::MatrixXd& Q = qr.matrixQ();
				const Eigen::Matrix<double, 9, 1>& null_space =
					Q.rightCols<1>();

				Model<3, 3> model;
				model.descriptor << null_space(0), null_space(1), null_space(2),
					null_space(3), null_space(4), null_space(5),
					null_space(6), null_space(7), null_space(8);
				models_.push_back(model);
				return true;
			}

			inline bool HomographyFourLineSolver::estimateNonMinimalModel(
				const std::vector<std::pair<size_t, size_t>>& sample_,
				std::vector<Model<3, 3>>& models_,
				const double* weights_) const
			{
				constexpr size_t equation_number = 2;
				const size_t& kSampleNumber = sample_.size();
				const size_t row_number = equation_number * kSampleNumber;
				Eigen::MatrixXd coefficients(row_number, 9);

				const size_t& kContainerIdx = sample_[0].first;
				const auto& kDataContainer = dataContainers[kContainerIdx];
				const size_t columns = kDataContainer->cols;
				const double* data_ptr = reinterpret_cast<double*>(kDataContainer->data);
				size_t row_idx = 0;
				double weight = 1.0;

				for (size_t i = 0; i < kSampleNumber; ++i)
				{
					const size_t &idx = sample_[i].second;

					const double* point_ptr =
						data_ptr + idx * columns;

					const double
						& x11 = point_ptr[0],
						& y11 = point_ptr[1],
						& x12 = point_ptr[2],
						& y12 = point_ptr[3],
						& x21 = point_ptr[4],
						& y21 = point_ptr[5],
						& x22 = point_ptr[6],
						& y22 = point_ptr[7];

					if (weights_ != nullptr)
						weight = weights_[idx];

					double a1, b1, c1,
						a2, b2, c2;

					getImplicitLine(x11, y11, x12, y12, a1, b1, c1);
					getImplicitLine(x21, y21, x22, y22, a2, b2, c2);

					double len1 = sqrt(a1 * a1 + b1 * b1),
						len2 = sqrt(a2 * a2 + b2 * b2);
					a1 /= len1;
					b1 /= len1;
					c1 /= len1;

					a2 /= len2;
					b2 /= len2;
					c2 /= len2;
					
					// TODO(danini): Add weighting
					coefficients(row_idx, 0) = -a2;
					coefficients(row_idx, 1) = 0;
					coefficients(row_idx, 2) = a1 * a2 / c1;
					coefficients(row_idx, 3) = -b2;
					coefficients(row_idx, 4) = 0;
					coefficients(row_idx, 5) = a1 * b2 / c1;
					coefficients(row_idx, 6) = -c2;
					coefficients(row_idx, 7) = 0;
					coefficients(row_idx, 8) = a1 / c1 * c2;
					++row_idx;

					coefficients(row_idx, 0) = 0;
					coefficients(row_idx, 1) = -a2;
					coefficients(row_idx, 2) = b1 * a2 / c1;
					coefficients(row_idx, 3) = 0;
					coefficients(row_idx, 4) = -b2;
					coefficients(row_idx, 5) = b1 * b2 / c1;
					coefficients(row_idx, 6) = 0;
					coefficients(row_idx, 7) = -c2;
					coefficients(row_idx, 8) = b1 / c1 * c2;
					++row_idx;
				}
				
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					coefficients,
					Eigen::ComputeFullV);
				const Eigen::Matrix<double, 9, 1>& null_space =
					svd.matrixV().rightCols<1>();

				Model<3, 3> model;
				model.descriptor << 
					null_space(0), null_space(1), null_space(2),
					null_space(3), null_space(4), null_space(5),
					null_space(6), null_space(7), null_space(8);
				models_.push_back(model);
				return true;
			}

			inline bool HomographyFourLineSolver::estimateModel(
				const std::vector<std::pair<size_t, size_t>>& sample_,
				std::vector<Model<3, 3>> &models_,
				const double *weights_) const
			{
				// Check if we are given enough points to fit the model
				assert(sample_.size() >= sampleSize());

				// If we have a minimal sample, it is usually enough to solve the problem with not necessarily
				// the most accurate solver. Therefore, we use normal equations for this
				return estimateNonMinimalModel(
					sample_,
					models_,
					weights_);
			}
		}
	}
}
