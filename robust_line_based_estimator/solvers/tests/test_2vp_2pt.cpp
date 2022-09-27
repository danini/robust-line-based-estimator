// 
// \author Petr Hruby
// \date May 2022
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

#include "solver_2vp_2pt.h"

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
		Eigen::Vector3d vps[2];
		Eigen::Vector3d vqs[2];
		Eigen::Vector3d pts[2];
		Eigen::Vector3d qts[2];
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d Tgt;
		sample(vps, vqs, pts, qts, Rgt, Tgt);
		
		//solve for the rotation
		Eigen::Matrix3d Rs[4];
		int sols = stage_1_solver_rotation_2vp(vps[0], vqs[0], vps[1], vqs[1], Rs);
		
		//for every rotation find the translation
		Eigen::Vector3d p1 = pts[0];
		Eigen::Vector3d q1 = qts[0];
		Eigen::Vector3d p2 = pts[1];
		Eigen::Vector3d q2 = qts[1];
		for(int j=0;j<sols;++j)
		{
			Eigen::Vector3d t;
			stage_2_solver_translation_2pt(p1, q1, p2, q2, Rs[j], t);
			
			std::cout << "Solution " << j << ". Rot err: " << evaluate_R(Rgt, Rs[j]) << ", Tr. err: " << evaluate_t(t, Tgt) << "\n\n";
		}
		std::cout << "\n";
	}

	return 0;
}
