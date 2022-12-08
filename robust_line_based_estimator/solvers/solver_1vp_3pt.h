// 
// \author Petr Hruby
// \date August 2022\
// #include <vector>
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>

//PROTOTYPES
//compute relative pose from 1 vanishing point and 3 points
inline int solver_wrapper_1vp_3pt(Eigen::Vector3d vp, //the vanishing point in 1st view
					Eigen::Vector3d vq, //the vanishing point in 2nd view
					Eigen::Vector3d * pts, //array of 3 points in the first view
					Eigen::Vector3d * qts, //array of 3 points in the second view
					Eigen::Matrix3d * Rs, //set of rotations consistent with the input data
					Eigen::Vector3d * Ts); //set of translation vectors consistent with the input data
				
//FUNCTIONS

inline int solver_wrapper_1vp_3pt(Eigen::Vector3d vp, Eigen::Vector3d vq, Eigen::Vector3d * pts, Eigen::Vector3d * qts, Eigen::Matrix3d * Rs, Eigen::Vector3d * Ts)
{
	vp = vp/vp.norm();
	vq = vq/vq.norm();
	
	//find the rotations that rotate vp, vq to y-axis
	const double d1 = std::sqrt(vp(0)*vp(0) + vp(2)*vp(2));
	Eigen::Vector3d axis1(-vp(2)/d1, 0, vp(0)/d1);
	const double ang1 = std::acos(vp(1));
	Eigen::Matrix3d A1x;
	A1x << 0, -axis1(2), axis1(1), axis1(2), 0, -axis1(0), -axis1(1), axis1(0), 0;
	Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity() + std::sin(ang1)*A1x + (1-std::cos(ang1))*A1x*A1x;

	const double d2 = std::sqrt(vq(0)*vq(0) + vq(2)*vq(2));
	Eigen::Vector3d axis2(-vq(2)/d2, 0, vq(0)/d2);
	const double ang2 = std::acos(vq(1));
	Eigen::Matrix3d A2x;
	A2x << 0, -axis2(2), axis2(1), axis2(2), 0, -axis2(0), -axis2(1), axis2(0), 0;
	Eigen::Matrix3d R2 = Eigen::Matrix3d::Identity() + std::sin(ang2)*A2x + (1-std::cos(ang2))*A2x*A2x;
	
	//also consider -vq (holds if one of the vanishing point is flipped)
	Eigen::Vector3d axis2f(vq(2)/d2, 0, -vq(0)/d2);
	const double ang2f = std::acos(-vq(1));
	Eigen::Matrix3d A2xf;
	A2xf << 0, -axis2f(2), axis2f(1), axis2f(2), 0, -axis2f(0), -axis2f(1), axis2f(0), 0;
	Eigen::Matrix3d R2f = Eigen::Matrix3d::Identity() + std::sin(ang2f)*A2xf + (1-std::cos(ang2f))*A2xf*A2xf;
	
	//rectify the projections p, q
	std::vector<Eigen::Vector3d> x1(3);
	std::vector<Eigen::Vector3d> x2(3);
	std::vector<Eigen::Vector3d> x2f(3);
	pose_lib::CameraPoseVector poses;
	pose_lib::CameraPoseVector poses_f;
	Eigen::Vector3d p[3];
	Eigen::Vector3d q[3];
	Eigen::Vector3d qf[3];
	for(int j=0;j<3;++j)
	{
		p[j] = R1*pts[j];
		q[j] = R2*qts[j];
		qf[j] = R2f*qts[j];
		
		p[j] = p[j]/p[j](2);
		q[j] = q[j]/q[j](2);
		qf[j] = qf[j]/qf[j](2);

		x1[j] = p[j];
		x2[j] = q[j];
		x2f[j] = qf[j];
	}

	int num_real_sols = 0;

	const int s1 = pose_lib::relpose_upright_3pt(x1, x2, &poses);
	for(int i=0;i<s1;++i)
	{
		Eigen::Matrix3d RR = poses[i].R;
		Eigen::Vector3d TR = poses[i].t;

		//find the rectified pose
		Eigen::Matrix3d R = R2.transpose()*RR*R1;
		Eigen::Vector3d T = R2.transpose()*TR;
			
		//store the solution
		Rs[num_real_sols] = R;
		Ts[num_real_sols] = T;
		++num_real_sols;
	}
	
	const int s2 = pose_lib::relpose_upright_3pt(x1, x2f, &poses_f);
	for(int i=0;i<s2;++i)
	{
		Eigen::Matrix3d RR = poses_f[i].R;
		Eigen::Vector3d TR = poses_f[i].t;

		//find the rectified pose
		Eigen::Matrix3d R = R2f.transpose()*RR*R1;
		Eigen::Vector3d T = R2f.transpose()*TR;
			
		//store the solution
		Rs[num_real_sols] = R;
		Ts[num_real_sols] = T;
		++num_real_sols;
	}

	return num_real_sols;
}


