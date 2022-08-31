#include <vector>	
#include <thread>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include "hybrid_ransac.h"
#include "statistics.h"

#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_homography_four_lines.h"
#include "estimators/homography_from_points_estimator.h"
#include "estimators/homography_from_lines_estimator.h"

#include "samplers/uniform_sampler.h"
#include "solver_selectors/random_solver_selector.h"
#include "scoring/model_verification.h"

#ifdef _WIN32
#include <direct.h>
#endif 

template<size_t _Rows, size_t _Columns>
void loadMatrix(
	const std::string& kFilename_,
	cv::Mat& container_);

void drawInliers(
	const cv::Mat& kImageSource_,
	const cv::Mat& kImageDestination_,
	const std::vector<const cv::Mat*>& kDataPoints_,
	const std::vector<std::pair<size_t, size_t>>& kInliers_,
	cv::Mat& outputImage_,
	const double kPointRadius = 5,
	const double kLineWidth = 2);

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));

	// Input data points
	cv::Mat pointCorrespondences,
		lineCorrespondences;

	loadMatrix<0, 4>("d:/Kutatas/PoseFromLineCorrespondences/robust-line-based-estimator/examples/data/point_matches.txt",
		pointCorrespondences);
	loadMatrix<0, 8>("d:/Kutatas/PoseFromLineCorrespondences/robust-line-based-estimator/examples/data/line_matches.txt",
		lineCorrespondences);

	cv::Mat imageSource = cv::imread("d:/Kutatas/PoseFromLineCorrespondences/robust-line-based-estimator/examples/data/terrace0.jpg"),
		imageDestination = cv::imread("d:/Kutatas/PoseFromLineCorrespondences/robust-line-based-estimator/examples/data/terrace1.jpg");

	// The data container containing pointers to all data types
	const std::vector<const cv::Mat *> kDataContainer = 
		{ &pointCorrespondences, &lineCorrespondences };

	// Include the namespace to keep the solver names shorter
	using namespace hybridransac::estimator::solver;
	using namespace hybridransac::estimator;
	using namespace hybridransac::solver_selector;
	using namespace hybridransac::sampler;
	using namespace hybridransac::scoring;

	// Class definitions that are used to define the estimators used and the residual functions for each data type
	using HomographyModel = hybridransac::Model<3, 3>; // The is the model type to be estimated. The template parameters are basically the <# rows, # columns> to represent the modl
	
	using EstimatorH4P = HomographyFromPointsEstimator< // The class for estimating homographies from point correspondences
		HomographyFourPointSolver, // The minimal solver for estimating a homography from point correspondences
		HomographyFourPointSolver, // The non-minimal solver for estimating a homography from point correspondences
		UniformSampler>; // The sampler that this solver uses to select minimal samples

	using EstimatorH4L = HomographyFromLinesEstimator< // The class for estimating homographies from line correspondences
		HomographyFourLineSolver, // The minimal solver for estimating a homography from line correspondences
		HomographyFourLineSolver, // The non-minimal solver for estimating a homography from line correspondences
		UniformSampler>; // The sampler that this solver uses to select minimal samples

	using SolverSelector = RandomSolverSelector< // The class that selects estimator to be used in the next iteration
		// This a parameter pack of the estimators that are used inside hybrid RANSAC. 
		// This should match with the estimators specified in hybrid RANSAC's parameter pack
		EstimatorH4P, 
		EstimatorH4L>;

	// Model verificator object to calculate the model score and select the inliers from heterogeneous data.
	using HomographyVerification = ModelVerification<
		HomographyModel, // The model to be estimated and verified
		// This a parameter pack of the scorings used for the data types. 
		// This does not have to match with the solvers of hybrid RANSAC.
		// For example, we might only want to use point correspondences for the verification and not lines.
		// In this particular example, we use both of them. 
		// Template parameters: 
		//		1. Estimator that knows the residual function, 
		//		2. Model to estimate, 
		//		3. Index of the data container in the container vector
		MSACScoringFunction<EstimatorH4P, HomographyModel, 0>, 
		MSACScoringFunction<EstimatorH4L, HomographyModel, 1>>;

	// Initializing a sampler for each data type
	// The sampler implements the access to the data
	// for fitting. In this example, use uniform samplers.
	UniformSampler pointSampler(
		// This vector should be of the same size as the data container. It specified how many points are needed from a particular container.
		// For example, in this case, 4 point and 0 line correspondences are selected.
		{ 4, 0 },
		// The data container vector to sample from
		kDataContainer);
	UniformSampler lineSampler(
		// This vector should be of the same size as the data container. It specified how many points are needed from a particular container.
		// For example, in this case, 0 point and 4 line correspondences are selected.
		{ 0, 4 },
		// The data container vector to sample from
		kDataContainer);

	// Initializing the solvers
	EstimatorH4P homography4P(&pointSampler, kDataContainer);
	EstimatorH4L homography4L(&lineSampler, kDataContainer);

	// Initializing the solver selector
	SolverSelector solverSelector;

	// Initializing the hybrid RANSAC object
	hybridransac::HybridRANSAC<
		cv::Mat, // The data container type
		HomographyModel, // The model to be estimated
		SolverSelector, // The solver selector type
		HomographyVerification, // The model verification type
		// The next types are a parameter pack. Arbitrary number of estimators
		// can be specified.
		EstimatorH4P,
		EstimatorH4L>
		hybridRANSAC;
	
	// Running hybrid RANSAC
	hybridRANSAC.run(
		kDataContainer, // The data container vector
		solverSelector, // The solver selector object
		// This is a parameter pack. The types of the parameters
		// should match with the types specified in the template.
		homography4P, 
		homography4L);

	// Print results
	const auto& statistics = hybridRANSAC.getStatistics();

	// Count the different inlier types
	std::vector<size_t> inlierTypeCounts(2, 0);
	for (const auto& [kTypeIdx, kPointIdx] : statistics.inliers)
		++inlierTypeCounts[kTypeIdx];

	printf("%d inliers are found out which\n", static_cast<int>(statistics.inliers.size()));
	printf("- %d are point correspondences\n", static_cast<int>(inlierTypeCounts[0]));
	printf("- %d are line correspondences\n", static_cast<int>(inlierTypeCounts[1]));
	printf("Iteration number = %d\n", static_cast<int>(statistics.iteration_number));
	printf("Processing time = %f\n", statistics.processing_time);

	// Draw the results
	cv::Mat outputImage;
	drawInliers(imageSource,
		imageDestination,
		kDataContainer,
		statistics.inliers,
		outputImage);

	cv::imshow("Found inliers", outputImage);
	cv::waitKey(0);

	return 0;
}

void drawInliers(
	const cv::Mat& kImageSource_,
	const cv::Mat& kImageDestination_,
	const std::vector<const cv::Mat *>& kDataPoints_,
	const std::vector<std::pair<size_t, size_t>> &kInliers_,
	cv::Mat &outputImage_,
	const double kPointRadius,
	const double kLineWidth)
{
	// Initialize the output image
	outputImage_.create(
		MAX(kImageSource_.rows, kImageDestination_.rows),
		kImageSource_.cols + kImageDestination_.cols,
		kImageSource_.type());

	// Copy the image contents to the output image
	cv::Rect roiSource(0, 0, kImageSource_.cols, kImageSource_.rows),
		roiDestination(kImageSource_.cols, 0, kImageDestination_.cols, kImageDestination_.rows);
	kImageSource_.copyTo(outputImage_(roiSource));
	kImageDestination_.copyTo(outputImage_(roiDestination));

	// Drawing the inliers to the image
	for (const auto& [kTypeIdx, kPointIdx] : kInliers_)
	{
		cv::Scalar color(
			255.0 * static_cast<double>(rand()) / RAND_MAX,
			255.0 * static_cast<double>(rand()) / RAND_MAX,
			255.0 * static_cast<double>(rand()) / RAND_MAX);

		// Selecting the current data point
		const cv::Mat& kDataPoint = 
			kDataPoints_[kTypeIdx]->row(kPointIdx);

		// Drawing point correspondence
		if (kTypeIdx == 0)
		{
			cv::circle(outputImage_,
				cv::Point(kDataPoint.at<double>(0), kDataPoint.at<double>(1)),
				kPointRadius,
				color,
				-1);

			cv::circle(outputImage_,
				cv::Point(kImageSource_.cols + kDataPoint.at<double>(2), kDataPoint.at<double>(3)),
				kPointRadius,
				color,
				-1);
		}
		else // Drawing line correspondence
		{
			cv::line(outputImage_,
				cv::Point(kDataPoint.at<double>(0), kDataPoint.at<double>(1)),
				cv::Point(kDataPoint.at<double>(2), kDataPoint.at<double>(3)),
				color,
				kLineWidth);

			cv::line(outputImage_,
				cv::Point(kImageSource_.cols + kDataPoint.at<double>(4), kDataPoint.at<double>(5)),
				cv::Point(kImageSource_.cols + kDataPoint.at<double>(6), kDataPoint.at<double>(7)),
				color,
				kLineWidth);
		}
	}
}

template<size_t _Rows, size_t _Columns>
void loadMatrix(
	const std::string& kFilename_,
	cv::Mat& container_)
{
	// Opening the file where the data is stored
	std::ifstream file(kFilename_);
	assert(file.is_open());

	size_t rowNumber = _Rows;
	if (_Rows == 0)
		file >> rowNumber;

	// Occupy the required memory
	container_.create(rowNumber, _Columns, CV_64F);

	// Get the pointer of the first element in the matrix
	double* dataPtr = reinterpret_cast<double*>(container_.data);

	// Load the data into the container
	while (file >> *(dataPtr++));

	// Close the input file
	file.close();
}