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

#include <vector>
#include <opencv2/core/core.hpp>
#include "uniform_random_generator.h"
#include "sampler.h"

namespace hybridransac
{
	namespace sampler
	{
		class UniformSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			std::vector<std::unique_ptr<utils::UniformRandomGenerator<size_t>>> 
				randomGenerators;
			std::vector<size_t> sampleIndices;

		public:
			explicit UniformSampler(
				const std::vector<size_t> &sampleSizes_,
				const std::vector<const cv::Mat*> &containers_)
				: Sampler(sampleSizes_, containers_)
			{
				randomGenerators.reserve(sampleSizes_.size());

				for (size_t samplerIdx = 0; samplerIdx < sampleSizes_.size(); ++samplerIdx)
				{
					randomGenerators.emplace_back(std::make_unique<utils::UniformRandomGenerator<size_t>>());
					randomGenerators.back()->resetGenerator(0,
						static_cast<size_t>(containers_[samplerIdx]->rows));
					sampleIndices.reserve(sampleIndices.size() + sampleSizes_[samplerIdx]);
				}
			}

			~UniformSampler()
			{
				for (auto& randomGenerator : randomGenerators)
					randomGenerator.release();
			}

			const std::string getName() const { return "Uniform Sampler"; }


			void update(
				const size_t* const subset_,
				const size_t& sample_size_,
				const size_t& iteration_number_,
				const double& inlier_ratio_)
			{

			}

			void reset()
			{
				for (size_t samplerIdx = 0; samplerIdx < sampleSizes.size(); ++samplerIdx)
				{
					randomGenerators.emplace_back(std::make_unique<utils::UniformRandomGenerator<size_t>>());
					randomGenerators.back()->resetGenerator(0,
						static_cast<size_t>(containers[samplerIdx]->rows));
				}
			}

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			inline bool sample(
				std::vector<std::pair<size_t, size_t>>& sample_);
		};

		inline bool UniformSampler::sample(
			std::vector<std::pair<size_t, size_t>> &sample_)
		{
			sample_.clear();
			for (size_t samplerIdx = 0; samplerIdx < sampleSizes.size(); ++samplerIdx)
			{
				const auto& sampleSize = sampleSizes[samplerIdx];
				if (sampleSize == 0)
					continue;

				sampleIndices.clear();
				randomGenerators[samplerIdx]->generateUniqueRandomSet(
					sampleIndices,
					sampleSize);

				for (const auto& idx : sampleIndices)
					sample_.emplace_back(std::make_pair(samplerIdx, idx));
			}
			return true;
		}
	}
}