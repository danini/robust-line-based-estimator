// 
// \author Petr Hruby
// \date September 2022
// 
#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>

//PROTOTYPES
//finds relative poses consistent with the input
inline int solver_1vp_2in(const Eigen::Vector3d vp, //vanishing point in the first camera
				const Eigen::Vector3d vq, //corresponding vanishing point in the second camera
				Eigen::Vector3d * ls, //line projections in the first camera: ls[0] and ls[1] intersect; ls[2] and ls[3] intersect; ls[0] is orthogonal to the vanishing point
				Eigen::Vector3d * ms, //corresponding vanishing point in the second camera
				Eigen::Matrix3d * Rs, //output rotations consistent with the input
				Eigen::Vector3d * Ts) //output translations consistent with the input

//FUNCTIONS
inline int solver_1vp_2in(const Eigen::Vector3d vp, const Eigen::Vector3d vq, Eigen::Vector3d * ls, Eigen::Vector3d * ms, Eigen::Matrix3d * Rs, Eigen::Vector3d * Ts)
{
	//find the rotations between the Manhattan world and each camera
	Eigen::Matrix3d R1;
	Eigen::Matrix3d R2;
	R1.col(0) = vp/vp.norm();
	Eigen::Vector3d r1_2 = vp.cross(ls[0]);
	R1.col(1) = r1_2/r1_2.norm();
	Eigen::Vector3d r1_3 = vp.cross(r1_2);
	R1.col(2) = r1_3/r1_3.norm();
	if(R1.determinant() < 0)
		R1.col(2) = -r1_3/r1_3.norm();
	
	R2.col(0) = vq/vq.norm();
	Eigen::Vector3d r2_2 = vq.cross(ms[0]);
	R2.col(1) = r2_2/r2_2.norm();
	Eigen::Vector3d r2_3 = vq.cross(r2_2);
	R2.col(2) = r2_3/r2_3.norm();
	if(R2.determinant() < 0)
		R2.col(2) = -r2_3/r2_3.norm();
		
	//find all possible rotations between the cameras
	Eigen::Matrix3d P2 = Eigen::Matrix3d::Identity();
	P2(1,1) = -1;
	P2(2,2) = -1;
	
	Eigen::Matrix3d P3 = Eigen::Matrix3d::Identity();
	P3(0,0) = -1;
	P3(2,2) = -1;
	
	Eigen::Matrix3d P4 = Eigen::Matrix3d::Identity();
	P4(0,0) = -1;
	P4(1,1) = -1;
	
	Rs[0] = R2*R1.transpose();
	Rs[1] = R2*P2*R1.transpose();
	Rs[2] = R2*P3*R1.transpose();
	Rs[3] = R2*P4*R1.transpose();
	
	//intersect the lines to get point correspondences (for the translation)
	Eigen::Vector3d p1 = ls[0].cross(ls[1]);
	p1 = p1/p1(2);
	Eigen::Vector3d q1 = ms[0].cross(ms[1]);
	q1 = q1/q1(2);
	
	Eigen::Vector3d p2 = ls[2].cross(ls[3]);
	p2 = p2/p2(2);
	Eigen::Vector3d q2 = ms[2].cross(ms[3]);
	q2 = q2/q2(2);
	
	//for every rotation find the translation
	for(int j=0;j<4;++j)
	{
		const Eigen::Matrix3d R = Rs[j];
	
		Eigen::Matrix<double,2,3> M;
		M(0,0) = p1(0)*q1(2)*R(1,0) + p1(1)*q1(2)*R(1,1) + p1(2)*q1(2)*R(1,2) - p1(0)*q1(1)*R(2,0) - p1(1)*q1(1)*R(2,1) - p1(2)*q1(1)*R(2,2);
		M(0,1) = -p1(0)*q1(2)*R(0,0) - p1(1)*q1(2)*R(0,1) - p1(2)*q1(2)*R(0,2) + p1(0)*q1(0)*R(2,0) + p1(1)*q1(0)*R(2,1) + p1(2)*q1(0)*R(2,2);
		M(0,2) = p1(0)*q1(1)*R(0,0) + p1(1)*q1(1)*R(0,1) + p1(2)*q1(1)*R(0,2) - p1(0)*q1(0)*R(1,0) - p1(1)*q1(0)*R(1,1) - p1(2)*q1(0)*R(1,2);
		M(1,0) = p2(0)*q2(2)*R(1,0) + p2(1)*q2(2)*R(1,1) + p2(2)*q2(2)*R(1,2) - p2(0)*q2(1)*R(2,0) - p2(1)*q2(1)*R(2,1) - p2(2)*q2(1)*R(2,2);
		M(1,1) = -p2(0)*q2(2)*R(0,0) - p2(1)*q2(2)*R(0,1) - p2(2)*q2(2)*R(0,2) + p2(0)*q2(0)*R(2,0) + p2(1)*q2(0)*R(2,1) + p2(2)*q2(0)*R(2,2);
		M(1,2) = p2(0)*q2(1)*R(0,0) + p2(1)*q2(1)*R(0,1) + p2(2)*q2(1)*R(0,2) - p2(0)*q2(0)*R(1,0) - p2(1)*q2(0)*R(1,1) - p2(2)*q2(0)*R(1,2);
		
		Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> USV(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = USV.matrixV();
		Eigen::Vector3d t = V.col(2);
		Ts[j] = t;
	}
	
	return 4;
}

