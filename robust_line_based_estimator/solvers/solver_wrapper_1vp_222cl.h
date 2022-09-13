// 
// \author Petr Hruby
// \date August 2022
// #include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "solver_1vp_222cl.cpp"

//PROTOTYPES

//samples projection of one vanishing point into two cameras whose relative pose is R,T
void sample_vp(const Eigen::Matrix3d R, //rotation between the cameras
		const Eigen::Vector3d T, //translation between the cameras
		Eigen::Vector3d &vp, //the vanishing point in the first camera
		Eigen::Vector3d &vq); //the vanishing point in the second camera

//samples one instance of the 1vp_222cl problem + its solution
void sample(Eigen::Vector3d &vp, //a vanishing point in the first camera
	Eigen::Vector3d &vq, //a vanishing point in the second camera
	Eigen::Vector3d * ls,  //array of projections of 6 lines into 1st camera; first two lines are coplanar, second two lines are coplanar, last two lines are coplanar
	Eigen::Vector3d * ms,  //array of projections of 6 lines into 2nd camera; first two lines are coplanar, second two lines are coplanar, last two lines are coplanar
	Eigen::Matrix3d &R, //rotation between the cameras
	Eigen::Vector3d &T); //translation between the cameras

//solves one instance of the 1vp_222cl problem
int solver_wrapper_1vp_222cl(Eigen::Vector3d vp, //a vanishing point in the first camera
				Eigen::Vector3d vq, //a vanishing point in the second camera
				Eigen::Vector3d * ls, //array of projections of 6 lines into 1st camera; first two lines are coplanar, second two lines are coplanar, last two lines are coplanar
				Eigen::Vector3d * ms, //array of projections of 6 lines into 2nd camera; first two lines are coplanar, second two lines are coplanar, last two lines are coplanar
				Eigen::Matrix3d * Rs, //rotations consistent with the input
				Eigen::Vector3d * Ts); //translations consistent with the input

//FUNCTIONS
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
	
	//GENERATE THE VANISHING POINT
	Eigen::Vector3d vp1;
	Eigen::Vector3d vq1;
	sample_vp(R, T, vp1, vq1);
	vp = vp1;
	vq = vq1;
	
	//generate 3 pairs of coplanar lines
	//3 points -> 1st pair; another 3 points -> 2nd pair, last 3 points -> 3rd pair
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
	
	double x4 = norm_sampler(eng);
	double y4 = norm_sampler(eng);
	double z4 = norm_sampler(eng);
	Eigen::Vector3d X4(x4, y4, z4+5);
	
	double x5 = norm_sampler(eng);
	double y5 = norm_sampler(eng);
	double z5 = norm_sampler(eng);
	Eigen::Vector3d X5(x5, y5, z5+5);
	
	double x6 = norm_sampler(eng);
	double y6 = norm_sampler(eng);
	double z6 = norm_sampler(eng);
	Eigen::Vector3d X6(x6, y6, z6+5);
	
	double x7 = norm_sampler(eng);
	double y7 = norm_sampler(eng);
	double z7 = norm_sampler(eng);
	Eigen::Vector3d X7(x7, y7, z7+5);
	
	double x8 = norm_sampler(eng);
	double y8 = norm_sampler(eng);
	double z8 = norm_sampler(eng);
	Eigen::Vector3d X8(x8, y8, z8+5);
	
	double x9 = norm_sampler(eng);
	double y9 = norm_sampler(eng);
	double z9 = norm_sampler(eng);
	Eigen::Vector3d X9(x9, y9, z9+5);
	
	//project the points into both cameras
	Eigen::Vector3d p1 = X1/X1(2);
	Eigen::Vector3d p2 = X2/X2(2);
	Eigen::Vector3d p3 = X3/X3(2);
	Eigen::Vector3d p4 = X4/X4(2);
	Eigen::Vector3d p5 = X5/X5(2);
	Eigen::Vector3d p6 = X6/X6(2);
	Eigen::Vector3d p7 = X7/X7(2);
	Eigen::Vector3d p8 = X8/X8(2);
	Eigen::Vector3d p9 = X9/X9(2);
	
	Eigen::Vector3d q1 = R*X1+T;
	q1 = q1/q1(2);
	Eigen::Vector3d q2 = R*X2+T;
	q2 = q2/q2(2);
	Eigen::Vector3d q3 = R*X3+T;
	q3 = q3/q3(2);
	Eigen::Vector3d q4 = R*X4+T;
	q4 = q4/q4(2);
	Eigen::Vector3d q5 = R*X5+T;
	q5 = q5/q5(2);
	Eigen::Vector3d q6 = R*X6+T;
	q6 = q6/q6(2);
	Eigen::Vector3d q7 = R*X7+T;
	q7 = q7/q7(2);
	Eigen::Vector3d q8 = R*X8+T;
	q8 = q8/q8(2);
	Eigen::Vector3d q9 = R*X9+T;
	q9 = q9/q9(2);
	
	//find the projections of the lines from the projections of the points
	Eigen::Vector3d l1 = p1.cross(p2);
	Eigen::Vector3d l2 = p1.cross(p3);
	
	Eigen::Vector3d l3 = p4.cross(p5);
	Eigen::Vector3d l4 = p4.cross(p6);
	
	Eigen::Vector3d l5 = p7.cross(p8);
	Eigen::Vector3d l6 = p7.cross(p9);
	
	Eigen::Vector3d m1 = q1.cross(q2);
	Eigen::Vector3d m2 = q1.cross(q3);
	
	Eigen::Vector3d m3 = q4.cross(q5);
	Eigen::Vector3d m4 = q4.cross(q6);
	
	Eigen::Vector3d m5 = q7.cross(q8);
	Eigen::Vector3d m6 = q7.cross(q9);
	
	ls[0] = l1;
	ls[1] = l2;
	ls[2] = l3;
	ls[3] = l4;
	ls[4] = l5;
	ls[5] = l6;
	
	ms[0] = m1;
	ms[1] = m2;
	ms[2] = m3;
	ms[3] = m4;
	ms[4] = m5;
	ms[5] = m6;
}


int solver_wrapper_1vp_222cl(Eigen::Vector3d vp, Eigen::Vector3d vq, Eigen::Vector3d * ls, Eigen::Vector3d * ms, Eigen::Matrix3d * Rs, Eigen::Vector3d * Ts)
{
	//normalize the vps
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
	
	//also try -vq (the vanishing point may be flipped)
	Eigen::Vector3d axis2f(vq(2)/d2, 0, -vq(0)/d2);
	const double ang2f = std::acos(-vq(1));
	Eigen::Matrix3d A2xf;
	A2xf << 0, -axis2f(2), axis2f(1), axis2f(2), 0, -axis2f(0), -axis2f(1), axis2f(0), 0;
	Eigen::Matrix3d R2f = Eigen::Matrix3d::Identity() + std::sin(ang2f)*A2xf + (1-std::cos(ang2f))*A2xf*A2xf;
	
	//rectify the line projections l, m
	Eigen::Vector3d l[6];
	Eigen::Vector3d m[6];
	Eigen::Vector3d mf[6];
	for(int j=0;j<6;++j)
	{
		l[j] = R1*ls[j];
		m[j] = R2*ms[j];
		mf[j] = R2f*ms[j];
		
		l[j] = l[j]/l[j].norm();
		m[j] = m[j]/m[j].norm();
		mf[j] = mf[j]/mf[j].norm();
	}
	
	//get the data vector: 6*l + 6*m for the original vanishing point
	Eigen::VectorXd data = Eigen::VectorXd(36);
	data.block<3,1>(0,0) = l[0];
	data.block<3,1>(3,0) = l[1];
	data.block<3,1>(6,0) = l[2];
	data.block<3,1>(9,0) = l[3];
	data.block<3,1>(12,0) = l[4];
	data.block<3,1>(15,0) = l[5];
	
	data.block<3,1>(18,0) = m[0];
	data.block<3,1>(21,0) = m[1];
	data.block<3,1>(24,0) = m[2];
	data.block<3,1>(27,0) = m[3];
	data.block<3,1>(30,0) = m[4];
	data.block<3,1>(33,0) = m[5];
	
	MatrixXcd t_sol = solver_1vp_222cl(data);
	
	int num_sols = 0;
	for(int i=0;i<54;++i)
	{
		if(t_sol(0,i).real() == 0) continue; //skip the complex solutions (that are actually set to zero by the automatically generated solver)
	
		//extract the rectified translation
		Eigen::Vector3d Tr = Eigen::Vector3d(t_sol(0,i).real(), t_sol(1,i).real(), 1);
		Tr = Tr/Tr.norm();
		
		//extract the rectified rotation
		double x = t_sol(2,i).real();
		Eigen::Matrix3d Rr = Eigen::Matrix3d::Identity();
		Rr(0,0) = (1-x*x)/(1+x*x);
		Rr(0,2) = (-2*x)/(1+x*x);
		Rr(2,0) = (2*x)/(1+x*x);
		Rr(2,2) = (1-x*x)/(1+x*x);
		
		//unrectify the rotation + the translation
		Eigen::Matrix3d R = R2.transpose()*Rr*R1;
		Eigen::Vector3d T = R2.transpose()*Tr;
		
		//store the rotation and translation
		Rs[num_sols] = R;
		Ts[num_sols] = T;
		++num_sols;
	}
	
	//get the data vector: 6*l + 6*m for the flipped vanishing point
	data.block<3,1>(18,0) = mf[0];
	data.block<3,1>(21,0) = mf[1];
	data.block<3,1>(24,0) = mf[2];
	data.block<3,1>(27,0) = mf[3];
	data.block<3,1>(30,0) = mf[4];
	data.block<3,1>(33,0) = mf[5];
	
	MatrixXcd tf_sol = solver_1vp_222cl(data);
	
	for(int i=0;i<54;++i)
	{
		if(tf_sol(0,i).real() == 0) continue; //skip the complex solutions (that are actually set to zero by the automatically generated solver)
	
		//extract the rectified translation
		Eigen::Vector3d Tr = Eigen::Vector3d(tf_sol(0,i).real(), tf_sol(1,i).real(), 1);
		Tr = Tr/Tr.norm();
		
		//extract the rectified rotation
		double x = tf_sol(2,i).real();
		Eigen::Matrix3d Rr = Eigen::Matrix3d::Identity();
		Rr(0,0) = (1-x*x)/(1+x*x);
		Rr(0,2) = (-2*x)/(1+x*x);
		Rr(2,0) = (2*x)/(1+x*x);
		Rr(2,2) = (1-x*x)/(1+x*x);
		
		//unrectify the rotation + the translation
		Eigen::Matrix3d R = R2f.transpose()*Rr*R1;
		Eigen::Vector3d T = R2f.transpose()*Tr;
		
		//store the rotation and translation
		Rs[num_sols] = R;
		Ts[num_sols] = T;
		++num_sols;
	}
	
	return num_sols;
}

