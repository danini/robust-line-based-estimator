// 
// \author Petr Hruby
// \date July 2022
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

#include "solver_2vp_cl_3stage.h"

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
		std::cout << "Problem " << i << "\n";
		//generate the simulated problem
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d Tgt;
		Eigen::Vector3d ngt;
		Eigen::Matrix3d Hgt;
		Eigen::Vector3d vp1;
		Eigen::Vector3d vq1;
		Eigen::Vector3d vp2;
		Eigen::Vector3d vq2;
		Eigen::Vector3d ls[3];
		Eigen::Vector3d ms[3];
		sample(Rgt, Tgt, ngt, Hgt, vp1, vq1, vp2, vq2, ls, ms);
		ngt = ngt/ngt(2);
		
		//stage 1: find the rotation matrix from the vanishing points
		Eigen::Matrix3d Rs[4];
		int sols = stage_1_solver_rotation_2vp(vp1, vq1, vp2, vq2, Rs);
		
		//check the found rotation matrices
		for(int j=0;j<sols;++j)
		{
			//stage 2: find the normal to the plane related by the homography
			Eigen::Vector3d n;
			stage_2_normal_R_vp_l(Rs[j], vp1, ls[0], ms[0], n);

			//stage 3: find the translation			
			Eigen::Vector3d t;
			stage_3_translation_3l_R_n(ls, ms, Rs[j], n, t);
			
			//std::cout << "Solution " << j << ". Rot err: " << evaluate_R(Rgt, Rs[j]) << ", Tr. err: " << evaluate_t(t, Tgt) << "\n\n";
			std::cout << "Solution " << j << ". Rot err: " << evaluate_R(Rgt, Rs[j]) << " Normal error: " << (n-ngt).norm() << ", Tr. err: " << evaluate_t(t, Tgt) << "\n\n";
		}
		std::cout << "\n";
	}

	return 0;
}
