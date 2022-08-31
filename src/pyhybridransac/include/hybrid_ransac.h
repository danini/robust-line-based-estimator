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

#include "settings.h"
#include "statistics.h"
#include "scoring/scoring_function.h"

namespace hybridransac
{
	template <
		class _ContainerType, // The type of the container being used to store the data. This currently is cv::Mat in most of the cases.
		class _Model, // The model type to be estimated
		class _SolverSelector, // The solver selector object that selects the solver to be used in the next iteration
		class _ModelVerification, // The model verification object that handles multiple data types
		class... _Estimators> // The parameter pack of the used estimators
		class HybridRANSAC
	{
	protected:
		// The number of estimators defined as a parameter pack
		static const std::size_t kEstimatorNumber = sizeof...(_Estimators);
		// Statistics object about the current run
		utils::Statistics statistics;
		// Settings object
		utils::Settings settings;

		// The function used for selectin the currently used estimator from
		// the parameter pack
		template <size_t I = 0>
		constexpr bool runEstimator(
			const size_t kSelectedEstimator_, // The index of the selected estimator
			std::tuple<_Estimators...>& kEstimators_, // All estimators used
			std::vector<_Model>& models_); // The estimated models

	public:
		HybridRANSAC()
		{
		}

		~HybridRANSAC() 
		{ 
		}

		// Returning the mutable settings object
		utils::Settings& getMutableSettings() { return settings; }
		// Returning the constant settings object
		const utils::Settings& getSettings() const { return settings; }
		// Returning the mutable settings object
		utils::Statistics& getMutableStatistics() { return statistics; }
		// Returning the constant settings object
		const utils::Statistics& getStatistics() const { return statistics; }

		// The main method applying Hybrid RANSAC to the input data points
		void run(
			const std::vector<const _ContainerType*>& data_, // The data used for the fitting
			_SolverSelector& solverSelector_, // The object selecting the next solver
			_Estimators... estimators_); // The parameter pack of the estimators used
	};

	template <
		class _ContainerType, // The type of the container being used to store the data. This currently is cv::Mat in most of the cases.
		class _Model, // The model type to be estimated
		class _SolverSelector, // The solver selector object that selects the solver to be used in the next iteration
		class _ModelVerification, // The model verification object that handles multiple data types
		class... _Estimators> // The parameter pack of the used estimators
	template <size_t I>
	constexpr bool HybridRANSAC<_ContainerType, _Model, _SolverSelector, _ModelVerification, _Estimators...>::runEstimator(
		const size_t kSelectedEstimator_, // The index of the selected estimator
		std::tuple<_Estimators...>& kEstimators_, // All estimators used
		std::vector<_Model>& models_) // The estimated models
	{
		// If we have iterated through all elements
		if constexpr (I == sizeof...(_Estimators))
		{
			// Last case, if nothing is left to
			// iterate, then exit the function
			return false;
		}
		else 
		{
			// This estimator is the currently selected one.
			if (kSelectedEstimator_ == I)
			{
				// Run the estimator that select a minimal sample and estimates the model
				// parameters from the selected sample.
				std::get<I>(kEstimators_).runMinimalSolver(models_);
				return models_.size() > 0;
			}

			// Going for next element.
			return runEstimator<I + 1>(kSelectedEstimator_, kEstimators_, models_);
		}
	}


	// The main method applying Graph-Cut RANSAC to the input data points_
	template <
		class _ContainerType, // The type of the container being used to store the data. This currently is cv::Mat in most of the cases.
		class _Model, // The model type to be estimated
		class _SolverSelector, // The solver selector object that selects the solver to be used in the next iteration
		class _ModelVerification, // The model verification object that handles multiple data types
		class... _Estimators> // The parameter pack of the used estimators
	void HybridRANSAC<_ContainerType, _Model, _SolverSelector, _ModelVerification, _Estimators...>::run(
		const std::vector<const _ContainerType*>& data_, // The data used for the fitting
		_SolverSelector& solverSelector_, // The object selecting the next solver
		_Estimators... estimators_) // The parameter pack of the estimators used
	{
		// Save the estimators as a parameter pack so they can be used later
		auto &estimators = ::std::make_tuple(::std::move(estimators_)...);

		// Initializing the model verification object
		_ModelVerification modelVerifier(settings.threshold);
	
		// Initializing the variables
		size_t iterationLimit = settings.max_iteration_number, // The current maximum number of iterations
			selectedSolver; // The index of the currently selected solver
		std::vector<_Model> models; // The estimated models
		_Model bestModel; // The best model so far
		scoring::Score currentScore, // The score of the current model
			bestScore; // The score of the best model so far

		// Variables for time measurement
		std::chrono::time_point<std::chrono::system_clock> start, end; 
		// The starting time of the neighborhood calculation
		start = std::chrono::system_clock::now(); 
		
		// Main RANSAC iteration that goes until the iteration number reaches its maximum
		while (statistics.iteration_number < iterationLimit)
		{
			// Increasing the iteration number
			++statistics.iteration_number;

			// Select the next solver to be used for estimating from a minimal sample
			selectedSolver = solverSelector_.selectNext();

			// Clear the model vector
			models.clear();

			// Apply the selected estimator.
			// This basically iterates through the parameter pack and runs
			// the solver specified as the selected one.
			runEstimator(
				selectedSolver, // The index of the selected solver
				estimators, // The parameter pack of the used estimators
				models); // The estimated models

			// Iterating through all models and check if one of the 
			// estimated models is better than the so-far-the-best one
			for (const auto& model : models)
			{
				// Resetting the current score as it will be newly calculated from the current model
				currentScore.reset();

				// Calculating the model score and selecting the inliers. 
				// This function iterates through multiple scoring functions and
				// data types to be able to select the inliers in a heterogeneous way.
				modelVerifier.getScore(
					data_, // The container of all data and data types used
					model, // The currently checked model
					currentScore); // The score of the currently checked model

				// If the current model's score is better than that of the so-far-the-best one
				if (bestScore < currentScore)
				{
					// Update the score of the best model
					bestScore.inlierNumber = currentScore.inlierNumber;
					bestScore.value = currentScore.value;
					bestScore.inliers.swap(currentScore.inliers);

					// Update the parameters of the best model
					bestModel = model;

					// Saving the type index that initialized the current model
					statistics.model_initializing_type = selectedSolver;

					// TODO: 
					// 1. Do local optimization in a heterogeneous way on all data types. Maybe with ceres?
					// 2. Adaptive iteration number
				}
			}
		}

		// The end time of the neighborhood calculation
		end = std::chrono::system_clock::now();
		// The elapsed time in seconds
		std::chrono::duration<double> elapsed_seconds = end - start; 

		// TODO: Run final fitting on all found inliers.
		statistics.processing_time = elapsed_seconds.count();
		statistics.inliers.swap(bestScore.inliers);
	}
}