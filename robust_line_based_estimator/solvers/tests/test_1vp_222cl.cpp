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

#include "solver_wrapper_1vp_222cl.h"


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
	
	return err;
}


int main(int argc, char **argv)
{	
	
	//experiments
	for(int i=0;i<100;++i)
	{
		//generate the simulated problem
		Eigen::Vector3d vp;
		Eigen::Vector3d vq;
		Eigen::Vector3d ls[6];
		Eigen::Vector3d ms[6];
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d Tgt;
		sample(vp, vq, ls, ms, Rgt, Tgt);
		//std::cout << Tgt << "\n\n";
		
		//find the solutions
		Eigen::Matrix3d Rs[108];
		Eigen::Vector3d Ts[108];
		int num_sols = solver_wrapper_1vp_222cl(vp, vq, ls, ms, Rs, Ts);
		
		//evaluate the solutions
		std::cout << "PROBLEM " << i << "\n";
		for(int j=0;j<num_sols;++j)
		{
			std::cout << "Solution " << j << ". Rot err: " << evaluate_R(Rgt, Rs[j]) << ", Tr. err: " << evaluate_t(Ts[j], Tgt) << "\n";
		}
		std::cout << "\n";
	}

	return 0;
}
