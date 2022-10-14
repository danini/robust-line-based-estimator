// 
// \author Petr Hruby
// \date June 2022
// #include <vector>
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>

//PROTOTYPES
//compute a rotation from two vanishing points
inline int stage_1_solver_rotation_2vp(Eigen::Vector3d vp1, //1st vanishing point in 1st view
								Eigen::Vector3d vq1, //1st vanishing point in 2nd view
								Eigen::Vector3d vp2, //2nd vanishing point in 1st view
								Eigen::Vector3d vq2, //2nd vanishing point in 2nd view
								Eigen::Matrix3d * Rs); //set of rotations consistent with the vanishing points (4 rotations)

//compute a translation from 2 points + known rotation
inline int stage_2_solver_translation_2pt(const Eigen::Vector3d p1, //1st point in 1st view
									const Eigen::Vector3d q1, //1st point in 2nd view
									const Eigen::Vector3d p2, //2nd point in 1st view
									const Eigen::Vector3d q2, //2nd point in 2nd view
									const Eigen::Matrix3d R, //rotation matrix
									Eigen::Vector3d &t); //translation vector

//FUNCTIONS
inline int stage_1_solver_rotation_2vp(Eigen::Vector3d vp1, Eigen::Vector3d vq1, Eigen::Vector3d vp2, Eigen::Vector3d vq2, Eigen::Matrix3d * Rs)
{
	//normalize the vps
	vp1 = vp1/vp1.norm();
	vq1 = vq1/vq1.norm();
	vp2 = vp2/vp2.norm();
	vq2 = vq2/vq2.norm();

	//solve for R
	Eigen::Matrix3d A;
	A.block<3,1>(0,0) = vp1;
	A.block<3,1>(0,1) = vp2;
	A.block<3,1>(0,2) = vp1.cross(vp2);
	Eigen::Matrix3d Ainv = A.inverse();
	
	Eigen::Matrix3d B1;
	B1.block<3,1>(0,0) = vq1;
	B1.block<3,1>(0,1) = vq2;
	B1.block<3,1>(0,2) = vq1.cross(vq2);
	
	Eigen::Matrix3d B2;
	B2.block<3,1>(0,0) = vq1;
	B2.block<3,1>(0,1) = -vq2;
	B2.block<3,1>(0,2) = vq1.cross(-vq2);
	
	Eigen::Matrix3d B3;
	B3.block<3,1>(0,0) = -vq1;
	B3.block<3,1>(0,1) = vq2;
	B3.block<3,1>(0,2) = -vq1.cross(vq2);
	
	Eigen::Matrix3d B4;
	B4.block<3,1>(0,0) = -vq1;
	B4.block<3,1>(0,1) = -vq2;
	B4.block<3,1>(0,2) = -vq1.cross(-vq2);
	
	Eigen::Matrix3d R1 = B1*Ainv;
	Eigen::Matrix3d R2 = B2*Ainv;
	Eigen::Matrix3d R3 = B3*Ainv;
	Eigen::Matrix3d R4 = B4*Ainv;
	Rs[0] = R1;
	Rs[1] = R2;
	Rs[2] = R3;
	Rs[3] = R4;
	
	return 4;
}

inline int stage_2_solver_translation_2pt(const Eigen::Vector3d p1, const Eigen::Vector3d q1, const Eigen::Vector3d p2, const Eigen::Vector3d q2, const Eigen::Matrix3d R, Eigen::Vector3d &t)
{
	Eigen::Matrix<double,2,3> M;
	M(0,0) = p1(0)*q1(2)*R(1,0) + p1(1)*q1(2)*R(1,1) + p1(2)*q1(2)*R(1,2) - p1(0)*q1(1)*R(2,0) - p1(1)*q1(1)*R(2,1) - p1(2)*q1(1)*R(2,2);
	M(0,1) = -p1(0)*q1(2)*R(0,0) - p1(1)*q1(2)*R(0,1) - p1(2)*q1(2)*R(0,2) + p1(0)*q1(0)*R(2,0) + p1(1)*q1(0)*R(2,1) + p1(2)*q1(0)*R(2,2);
	M(0,2) = p1(0)*q1(1)*R(0,0) + p1(1)*q1(1)*R(0,1) + p1(2)*q1(1)*R(0,2) - p1(0)*q1(0)*R(1,0) - p1(1)*q1(0)*R(1,1) - p1(2)*q1(0)*R(1,2);
	M(1,0) = p2(0)*q2(2)*R(1,0) + p2(1)*q2(2)*R(1,1) + p2(2)*q2(2)*R(1,2) - p2(0)*q2(1)*R(2,0) - p2(1)*q2(1)*R(2,1) - p2(2)*q2(1)*R(2,2);
	M(1,1) = -p2(0)*q2(2)*R(0,0) - p2(1)*q2(2)*R(0,1) - p2(2)*q2(2)*R(0,2) + p2(0)*q2(0)*R(2,0) + p2(1)*q2(0)*R(2,1) + p2(2)*q2(0)*R(2,2);
	M(1,2) = p2(0)*q2(1)*R(0,0) + p2(1)*q2(1)*R(0,1) + p2(2)*q2(1)*R(0,2) - p2(0)*q2(0)*R(1,0) - p2(1)*q2(0)*R(1,1) - p2(2)*q2(0)*R(1,2);
	
	Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> USV(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d V = USV.matrixV();
	t = V.col(2);
	return 1;
}

