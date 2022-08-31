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

#include <math.h>
#include <random>
#include <unordered_set>
#include <vector>
#include <opencv2/core.hpp>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

namespace hybridransac
{
	namespace scoring
	{
		/* RANSAC Scoring */
		struct Score {

			/* Number of inliers, rectangular gain function */
			size_t inlierNumber;

			/* The inliers of the current model */
			std::vector<std::pair<size_t, size_t>> inliers;

			/* Score */
			double value;

			Score() :
				inlierNumber(0),
				value(0.0)
			{

			}

			void reset()
			{
				inlierNumber = 0;
				inliers.clear();
				value = 0;
			}

			inline bool operator<(const Score& score_)
			{
				return value < score_.value;
			}

			inline bool operator>(const Score& score_)
			{
				return *this > score_;
			}
		};

		template<class _ModelEstimator,
			class _ModelType,
			size_t _ContainerIndex>
		class ScoringFunction
		{
		public:
			ScoringFunction()
			{

			}

			virtual ~ScoringFunction()
			{

			}

			/*static bool getScore(const cv::Mat& points_, // The input data points
				_Model& model_/*, // The current model parameters
				const _ModelEstimator& estimator_, // The model estimator
				const double threshold_, // The inlier-outlier threshold
				std::vector<size_t>& inliers_, // The selected inliers
				const Score& best_score_ = Score(), // The score of the current so-far-the-best model
				const bool store_inliers_ = true) = 0;*/  // A flag to decide if the inliers should be stored

		};

		template<class _ModelEstimator,
			class _ModelType,
			size_t _ContainerIndex>
		class MSACScoringFunction : public ScoringFunction<_ModelEstimator, _ModelType, _ContainerIndex>
		{
		protected:

		public:
			MSACScoringFunction()
			{

			}

			~MSACScoringFunction()
			{

			}

			static size_t getContainerIdx() { return _ContainerIndex; }

			// Return the score of a model w.r.t. the data points and the threshold
			static bool getScore(const cv::Mat& kPoints_, // The input data points
				const _ModelType& kModel_,
				const double kSquaredThreshold_,
				const size_t kTypeIdx_,
				Score& currentScore_
				/*, // The current model parameters
				const _ModelEstimator& estimator_, // The model estimator
				const double threshold_, // The inlier-outlier threshold
				std::vector<size_t>& inliers_, // The selected inliers
				const Score& best_score_ = Score(), // The score of the current so-far-the-best model
				const bool store_inliers_ = true*/) // A flag to decide if the inliers should be stored
			{
				double squaredResidual;
				const size_t &kPointNumber = kPoints_.rows;

				// Iterate through all points, calculate the squared_residuals and store the points as inliers if needed.
				for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
				{
					// Calculate the point-to-model residual
					squaredResidual =
						_ModelEstimator::squaredResidual(kPoints_.row(pointIdx),
							kModel_);

					// If the residual is smaller than the threshold, store it as an inlier and
					// increase the score.
					if (squaredResidual < kSquaredThreshold_)
					{
						// Store the point as an inlier.
						currentScore_.inliers.emplace_back(std::make_pair(kTypeIdx_, pointIdx));

						// Increase the inlier number
						++(currentScore_.inlierNumber);
						// Increase the score. The original truncated quadratic loss is as follows: 
						// 1 - residual^2 / threshold^2. For RANSAC, -residual^2 is enough.
						// It has been re-arranged as
						// score = 1 - residual^2 / threshold^2				->
						// score threshold^2 = threshold^2 - residual^2		->
						// score threshold^2 - threshold^2 = - residual^2.
						// This is faster to calculate and it is normalized back afterwards.
						//currentScore_.value -= squaredResidual; // Truncated quadratic cost
						currentScore_.value += 1.0 - squaredResidual / kSquaredThreshold_; // Truncated quadratic cost
					}

					// Interrupt if there is no chance of being better than the best model
					//if (kPointNumber - pointIdx - currentScore_.value < -best_score_.value)
					//	return false;
				}

				if (currentScore_.inlierNumber == 0)
					return false;

				// Return the final score
				return true;
			}
		};
	}
}