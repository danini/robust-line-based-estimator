// 
// \author Petr Hruby
// \date September 2022
// 
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>
#include <string>
#include <random>

#include "solver_1vp_2in_orthogonal.h"

using namespace std::chrono;

void sample_vp(const Eigen::Matrix3d R, const Eigen::Vector3d T, Eigen::Vector3d &vp, Eigen::Vector3d &vq)
{
	//initialize the random samplers
	std::normal_distribution<double> norm_sampler(0.0,1.0);
	std::random_device rd;
    std::default_random_engine eng(rd());

	//generate a direction of a line and two initial points
	double dir1_x = norm_sampler(eng);
	double dir1_y = norm_sampler(eng);
	double dir1_z = norm_sampler(eng);
	Eigen::Vector3d dir1(dir1_x, dir1_y, dir1_z);
	
	double x01 = norm_sampler(eng);
	double y01 = norm_sampler(eng);
	double z01 = norm_sampler(eng);
	Eigen::Vector3d X01(x01, y01, z01+5);
	
	double x02 = norm_sampler(eng);
	double y02 = norm_sampler(eng);
	double z02 = norm_sampler(eng);
	Eigen::Vector3d X02(x02, y02, z02+5);
	
	//get the points further on the line
	Eigen::Vector3d X11 = X01+dir1;
	Eigen::Vector3d X12 = X02+dir1;
	
	//project the points to the cameras
	Eigen::Vector3d p01 = X01/X01(2);
	Eigen::Vector3d p11 = X11/X11(2);
	Eigen::Vector3d p02 = X02/X02(2);
	Eigen::Vector3d p12 = X12/X12(2);
	
	Eigen::Vector3d q01 = R*X01+T;
	q01 = q01/q01(2);
	Eigen::Vector3d q11 = R*X11+T;
	q11 = q11/q11(2);
	Eigen::Vector3d q02 = R*X02+T;
	q02 = q02/q02(2);
	Eigen::Vector3d q12 = R*X12+T;
	q12 = q12/q12(2);
	
	//construct the projected lines and extract the vanishing points
	Eigen::Vector3d l1 = p01.cross(p11);
	Eigen::Vector3d l2 = p02.cross(p12);
	vp = l1.cross(l2);
	
	Eigen::Vector3d m1 = q01.cross(q11);
	Eigen::Vector3d m2 = q02.cross(q12);
	vq = m1.cross(m2);
}

void sample(Eigen::Vector3d &vp, Eigen::Vector3d &vq, Eigen::Vector3d * ls, Eigen::Vector3d * ms, Eigen::Matrix3d &R, Eigen::Vector3d &T)
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
	
	//GENERATE 1 VANISHING POINT
	Eigen::Vector3d vp1;
	Eigen::Vector3d vq1;
	sample_vp(R, T, vp1, vq1);
	vp = vp1;
	vq = vq1;
	
	//GENERATE A DIRECTION ORTHOGONAL TO THE VP
	Eigen::Vector3d u(1,1,1);
	Eigen::Vector3d dir = vp1.cross(u);	
	
	//GENERATE 2 FINITE POINTS + for each point 4 endpoints of lines intersecting in the points (mean [0 0 5], std 1)
	double x1 = norm_sampler(eng);
	double y1 = norm_sampler(eng);
	double z1 = norm_sampler(eng);
	Eigen::Vector3d X1(x1, y1, z1+5);
	
	Eigen::Vector3d X2a = X1+dir;
	Eigen::Vector3d X2b = X1-dir;
	
	double x3 = norm_sampler(eng);
	double y3 = norm_sampler(eng);
	double z3 = norm_sampler(eng);
	Eigen::Vector3d X3a(x3, y3, z3+5);
	Eigen::Vector3d X3b = X1 - (X3a-X1);
	
	double x4 = norm_sampler(eng);
	double y4 = norm_sampler(eng);
	double z4 = norm_sampler(eng);
	Eigen::Vector3d X4(x4, y4, z4+5);
	
	double x5 = norm_sampler(eng);
	double y5 = norm_sampler(eng);
	double z5 = norm_sampler(eng);
	Eigen::Vector3d X5a(x5, y5, z5+5);
	Eigen::Vector3d X5b = X4 - (X5a-X4);
	
	double x6 = norm_sampler(eng);
	double y6 = norm_sampler(eng);
	double z6 = norm_sampler(eng);
	Eigen::Vector3d X6a(x6, y6, z6+5);
	Eigen::Vector3d X6b = X4 - (X6a-X4);
	
	//project the original points (for check)
	/*Eigen::Vector3d p1 = X1/X1(2);
	Eigen::Vector3d q1 = R*X1+T;
	q1 = q1/q1(2);
	
	Eigen::Vector3d p4 = X4/X4(2);
	Eigen::Vector3d q4 = R*X4+T;
	q4 = q4/q4(2);*/
	
	//project all the endpoints
	Eigen::Vector3d p2a = X2a/X2a(2);
	Eigen::Vector3d q2a = R*X2a+T;
	q2a = q2a/q2a(2);
	
	Eigen::Vector3d p2b = X2b/X2b(2);
	Eigen::Vector3d q2b = R*X2b+T;
	q2b = q2b/q2b(2);
	
	Eigen::Vector3d p3a = X3a/X3a(2);
	Eigen::Vector3d q3a = R*X3a+T;
	q3a = q3a/q3a(2);
	
	Eigen::Vector3d p3b = X3b/X3b(2);
	Eigen::Vector3d q3b = R*X3b+T;
	q3b = q3b/q3b(2);
	
	//
	Eigen::Vector3d p5a = X5a/X5a(2);
	Eigen::Vector3d q5a = R*X5a+T;
	q5a = q5a/q5a(2);
	
	Eigen::Vector3d p5b = X5b/X5b(2);
	Eigen::Vector3d q5b = R*X5b+T;
	q5b = q5b/q5b(2);
	
	Eigen::Vector3d p6a = X6a/X6a(2);
	Eigen::Vector3d q6a = R*X6a+T;
	q6a = q6a/q6a(2);
	
	Eigen::Vector3d p6b = X6b/X6b(2);
	Eigen::Vector3d q6b = R*X6b+T;
	q6b = q6b/q6b(2);
	
	//compute the lines from the endpoints
	Eigen::Vector3d l2 = p2a.cross(p2b);
	Eigen::Vector3d m2 = q2a.cross(q2b);
	
	Eigen::Vector3d l3 = p3a.cross(p3b);
	Eigen::Vector3d m3 = q3a.cross(q3b);
	
	//
	Eigen::Vector3d l5 = p5a.cross(p5b);
	Eigen::Vector3d m5 = q5a.cross(q5b);
	
	Eigen::Vector3d l6 = p6a.cross(p6b);
	Eigen::Vector3d m6 = q6a.cross(q6b);
	
	//store the lines
	ls[0] = l2;
	ls[1] = l3;
	ls[2] = l5;
	ls[3] = l6;
	
	ms[0] = m2;
	ms[1] = m3;
	ms[2] = m5;
	ms[3] = m6;
}

double evaluate_R(Eigen::Matrix3d R, Eigen::Matrix3d gtR)
{
	Eigen::Matrix3d R_diff = R * gtR.transpose();
	double cos = (R_diff.trace()-1)/2;
	double err = std::acos(cos);
	if(cos > 1)
		err = 0;
		
	return err;
}

double evaluate_t(Eigen::Vector3d t, Eigen::Vector3d gtT)
{
	double n1 = t.norm();
	double n2 = gtT.norm();
	double cos = (gtT.transpose() * t);
	cos = cos/(n1*n2);
	cos = std::abs(cos);
	double err = std::acos(cos);
	if(cos > 1)
		err = 0;
	if(cos < -1)
		err = 3.14;
	
	return err;
}

int main(int argc, char **argv)
{	
	//experiments
	for(int i=0;i<100;++i)
	{
		std::cout << "Problem " << i << "\n";
		//generate the simulated problem
		Eigen::Vector3d vp;
		Eigen::Vector3d vq;
		Eigen::Vector3d ls[4];
		Eigen::Vector3d ms[4];
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d Tgt;
		sample(vp, vq, ls, ms, Rgt, Tgt);
		
		//solve for the relative pose
		Eigen::Matrix3d Rs[4];
		Eigen::Vector3d Ts[4];
		const int num_sols = solver_1vp_2in(vp, vq, ls, ms, Rs, Ts);
		
		//check the solutions
		for(int j=0;j<num_sols;++j)
		{
			std::cout << "Solution " << j << ". Rot err: " << evaluate_R(Rgt, Rs[j]) << ", Tr. err: " << evaluate_t(Ts[j], Tgt) << "\n\n";
		}
		std::cout << "\n";
	}

	return 0;
}
