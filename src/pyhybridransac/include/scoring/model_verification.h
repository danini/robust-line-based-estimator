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

#include "scoring_function.h"

namespace hybridransac
{
	namespace scoring
	{
		template <typename _ModelType,
			typename ..._ScoringFunctions>
			class ModelVerification
		{
		protected:
			// The number of estimators defined as a parameter pack
			static const std::size_t kEstimatorNumber = sizeof...(_ScoringFunctions);

			const double kInlierOutlierThreshold,
				kSquaredThreshold;

		public:
			ModelVerification(const double kInlierOutlierThreshold_) : 
				kInlierOutlierThreshold(kInlierOutlierThreshold_),
				kSquaredThreshold(kInlierOutlierThreshold_ * kInlierOutlierThreshold_)
			{
			}
			~ModelVerification() {}

			bool getScore(
				const std::vector<const cv::Mat*>& data_, // The data used for the fitting
				const _ModelType& model_,
				Score& score_) const;
		};

		template <typename _ModelType,
			typename ..._ScoringFunctions>
			bool ModelVerification<_ModelType, _ScoringFunctions...>::getScore(
				const std::vector<const cv::Mat*>& data_, // The data used for the fitting
				const _ModelType& model_,
				Score &score_) const
		{
			bool successes[] = { (_ScoringFunctions::getScore(
				*data_[_ScoringFunctions::getContainerIdx()], 
				model_,
				kSquaredThreshold,
				_ScoringFunctions::getContainerIdx(),
				score_))...};

			bool success = false;
			for (size_t estimatorIdx = 0; estimatorIdx < kEstimatorNumber; ++estimatorIdx)
				success = success || successes[estimatorIdx];
			return success;
		}
	}
}