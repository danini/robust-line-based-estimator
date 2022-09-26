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

#include "helper_functions_2vp_3cl.h"


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
		std::cout << "PROBLEM " << i << "\n";
		//generate the simulated problem
		//simulate vps, vqs, ls, ms, R, t 
		Eigen::Vector3d vps[2];
		Eigen::Vector3d vqs[2];
		Eigen::Vector3d ls[3];
		Eigen::Vector3d ms[3];
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d Tgt;
		sample(vps, vqs, ls, ms, Rgt, Tgt);
		
		//stage 1
		//solve for the rotation
		Eigen::Matrix3d Rs[4];
		int sols = stage_1_solver_rotation_2vp(vps[0], vqs[0], vps[1], vqs[1], Rs);
		
		//stage 2
		//numerical conditioning, TODO is this good to do?
		Eigen::Vector3d l1 = ls[0]/ls[0].norm();
		Eigen::Vector3d m1 = ms[0]/ms[0].norm();
		Eigen::Vector3d l2 = ls[1]/ls[1].norm();
		Eigen::Vector3d m2 = ms[1]/ms[1].norm();
		Eigen::Vector3d l3 = ls[2]/ls[2].norm();
		Eigen::Vector3d m3 = ms[2]/ms[2].norm();
		/*Eigen::Vector3d l1 = ls[0];
		Eigen::Vector3d m1 = ms[0];
		Eigen::Vector3d l2 = ls[1];
		Eigen::Vector3d m2 = ms[1];
		Eigen::Vector3d l3 = ls[2];
		Eigen::Vector3d m3 = ms[2];*/
		
		for(int j=0;j<4;++j)
		{
			Eigen::Matrix3d R = Rs[j];
			
			//solve for the translation
			Eigen::MatrixXcd t_sol = stage_2_wrapper_translation_3cl(l1, m1, l2, m2, l3, m3, R);		
			
			//evaluate the results
			for(int k=0;k<6;++k)
			{
				Eigen::Vector3d t = t_sol.block<2,1>(0,k).real().homogeneous();
				std::cout << "Solution: (" << j << ", " << k << "); Rot. err.: " << evaluate_R(R, Rgt) << ", Tr. err.: " << evaluate_t(t, Tgt) << "\n";
			}
		}
		std::cout << "\n\n";
		
	}

	return 0;
}
