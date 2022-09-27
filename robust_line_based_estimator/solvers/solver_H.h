// 
// \author Petr Hruby
// \date July 2022
// 

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>
#include <random>

#include "5pt_solver.hxx" // extracted from COLMAP, not the fastest solution

//PROTOTYPES
//finds a set of essential matrices that are consistent with the input homography
int solve_homography_calibrated(const Eigen::Matrix3d H, //homography relating 2 calibrated views
								Eigen::Matrix3d * Es); //output set of essential matrices

//sample a relative pose + one calibrated homography between the views
void sample_homography_calibrated(Eigen::Matrix3d &H, //homography relating 2 calibrated views
								Eigen::Matrix3d &R, //ground truth rotation consistent with the homography
								Eigen::Vector3d &T); //ground truth translation consistent with the homography

//FUNCTIONS
int solve_homography_calibrated(const Eigen::Matrix3d H, Eigen::Matrix3d * Es)
{
	//initialize the random samplers
	std::normal_distribution<double> norm_sampler(0.0,1.0);
	std::random_device rd;
    std::default_random_engine eng(rd());
    
    Eigen::Matrix<double,5,9> AA;

	//generate 5 random 2D points in the first camera
	Eigen::Vector2d points1[5];
	Eigen::Vector2d points2[5];
	for(int i=0;i<5;++i)
	{
		double x = norm_sampler(eng);
		double y = norm_sampler(eng);
		Eigen::Vector3d p(x, y, 1);
		
		//use the homography to get the counterpart of the points in the 2nd camera
		Eigen::Vector3d q = H*p;
		q = q/q(2);
		
		Eigen::Vector2d pp(x,y);
		Eigen::Vector2d qq(q(0),q(1));
		
		points1[i] = pp;
		points2[i] = qq;
		
		//get a linear constraint used in the computation of the essential matrix and put it to one row of a matrix AA (to check how well the system is conditioned)
		AA(i,0) = p(0)*q(0);
		AA(i,1) = p(0)*q(1);
		AA(i,2) = p(0)*q(2);
		
		AA(i,3) = p(1)*q(0);
		AA(i,4) = p(1)*q(1);
		AA(i,5) = p(1)*q(2);
		
		AA(i,6) = p(2)*q(0);
		AA(i,7) = p(2)*q(1);
		AA(i,8) = p(2)*q(2);
	}
	
	//use SVD to check how well conditioned the system is, if the conditioning is bad, resample the points
	//TODO use the arrays points1 + points2 and compactify this, and move after the generation of the points (together with definition of AA)
	//TODO how to generate 5 well conditioned samples?
	Eigen::JacobiSVD<Eigen::Matrix<double, 5, 9>> svd(AA, Eigen::ComputeFullU | Eigen::ComputeFullV);
	std::cout << svd.singularValues() << "\n\n";
	
	int sols = essential_solver(points1, points2, Es);
	
	//THIS REDUCES THE NUMBER OF SOLUTIONS TO TWO
	//generate the 6th point to select the correct essential matrix
	/*double x6 = norm_sampler(eng);
	double y6 = norm_sampler(eng);
	Eigen::Vector3d p6(x6, y6, 1);
	Eigen::Vector3d q6 = H*p6;
	q6 = q6/q6(2);
	for(int i=0;i<sols;++i)
	{
		std::cout << "Sol: " << i << ", 6th pt error: " << q6.transpose() * Es[i] * p6 << "\n";
	}*/
	
	return sols;
}

void sample_homography_calibrated(Eigen::Matrix3d &H, Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
	//init the random samplers
	std::normal_distribution<double> norm_sampler(0.0,1.0);
	std::uniform_real_distribution<double> ax_dir_sampler(-3.141592654, 3.141592654);
	std::uniform_real_distribution<double> z_sampler(-1.0, 1.0);
	std::random_device rd;
    std::default_random_engine eng(rd());
    
	//GENERATE THE RELATIVE POSE
	//generate the center of the 2nd camera
	double axC2 = ax_dir_sampler(eng);
	double zC2 = z_sampler(eng);
	double normC2 = norm_sampler(eng);
	Eigen::Vector3d C2(std::sqrt(1-zC2*zC2)*std::cos(axC2), std::sqrt(1-zC2*zC2)*std::sin(axC2), zC2);
	C2 = C2/C2.norm();
	C2 = normC2*C2;
	
	//generate the angle of the 2nd rotation and build the rotation matrix and the translation vector
	double alpha_x = norm_sampler(eng);
	Eigen::Matrix3d Rx;
	Rx << 1,0,0, 0, std::cos(alpha_x), -std::sin(alpha_x), 0, std::sin(alpha_x), std::cos(alpha_x);
	double alpha_y = norm_sampler(eng);
	Eigen::Matrix3d Ry;
	Ry << std::cos(alpha_y), 0, -std::sin(alpha_y), 0, 1, 0, std::sin(alpha_y), 0, std::cos(alpha_y);
	double alpha_z = norm_sampler(eng);
	Eigen::Matrix3d Rz;
	Rz << std::cos(alpha_y), -std::sin(alpha_y), 0, std::sin(alpha_y), std::cos(alpha_y), 0, 0,0,1;
	R = Rx*Ry*Rz;
	T = -R*C2;
	
	//GENERATE A HOMOGRAPHY
	//generate 4 points on a plane
	double x1 = norm_sampler(eng);
	double y1 = norm_sampler(eng);
	double z1 = norm_sampler(eng);
	Eigen::Vector3d X1(x1, y1, z1+5);
	
	double x2 = norm_sampler(eng);
	double y2 = norm_sampler(eng);
	double z2 = norm_sampler(eng);
	Eigen::Vector3d X2(x2, y2, z2+5);
	
	double x3 = norm_sampler(eng);
	double y3 = norm_sampler(eng);
	double z3 = norm_sampler(eng);
	Eigen::Vector3d X3(x3, y3, z3+5);
	
	//generate 4th point on the plane defined by the 3 points
	double a = norm_sampler(eng);
	double b = norm_sampler(eng);
	Eigen::Vector3d X4 = X1 + a*(X2-X1) + b*(X3-X1);
	
	//project the points into the cameras
	Eigen::Vector3d p1 = X1/X1(2);
	Eigen::Vector3d p2 = X2/X2(2);
	Eigen::Vector3d p3 = X3/X3(2);
	Eigen::Vector3d p4 = X4/X4(2);
	
	Eigen::Vector3d q1 = R*X1+T;
	q1 = q1/q1(2);
	Eigen::Vector3d q2 = R*X2+T;
	q2 = q2/q2(2);
	Eigen::Vector3d q3 = R*X3+T;
	q3 = q3/q3(2);
	Eigen::Vector3d q4 = R*X4+T;
	q4 = q4/q4(2);
	
	//compute the homography from the points
	Eigen::Matrix<double, 8, 9> M = Eigen::Matrix<double, 8, 9>::Zero(8,9);
	M.block<1,3>(0,0) = p1.transpose();
	M.block<1,3>(0,6) = -q1(0)*p1.transpose();
	M.block<1,3>(1,3) = p1.transpose();
	M.block<1,3>(1,6) = -q1(1)*p1.transpose();
	
	M.block<1,3>(2,0) = p2.transpose();
	M.block<1,3>(2,6) = -q2(0)*p2.transpose();
	M.block<1,3>(3,3) = p2.transpose();
	M.block<1,3>(3,6) = -q2(1)*p2.transpose();
	
	M.block<1,3>(4,0) = p3.transpose();
	M.block<1,3>(4,6) = -q3(0)*p3.transpose();
	M.block<1,3>(5,3) = p3.transpose();
	M.block<1,3>(5,6) = -q3(1)*p3.transpose();
	
	M.block<1,3>(6,0) = p4.transpose();
	M.block<1,3>(6,6) = -q4(0)*p4.transpose();
	M.block<1,3>(7,3) = p4.transpose();
	M.block<1,3>(7,6) = -q4(1)*p4.transpose();
	
	Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> USV(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXd V = USV.matrixV();
	Eigen::VectorXd h = V.col(8);
	H.block<1,3>(0,0) = h.block<3,1>(0,0).transpose();
	H.block<1,3>(1,0) = h.block<3,1>(3,0).transpose();
	H.block<1,3>(2,0) = h.block<3,1>(6,0).transpose();
}
