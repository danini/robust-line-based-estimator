// 
// \author Petr Hruby
// \date August 2022
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

#include "solver_1vp_3pt.h"

using namespace std::chrono;

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
		Eigen::Vector3d pts[3];
		Eigen::Vector3d qts[3];
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d Tgt;
		sample(&vp, &vq, pts, qts, Rgt, Tgt);	
		
		//there are 2 feasible 
		//TODO see if they yield the same solutions
		
		//solve for the relative pose
		Eigen::Matrix3d Rs[12];
		Eigen::Vector3d ts[12];
		int sols = solver_wrapper_1vp_3pt(vp, vq, pts, qts, Rs, ts);
		
		//evaluate the obtained solutions
		for(int j=0;j<sols;++j)
		{
			std::cout << "Solution " << j << ". Rot err: " << evaluate_R(Rgt, Rs[j]) << ", Tr. err: " << evaluate_t(ts[j], Tgt) << "\n\n";
		}
	}

	return 0;
}
