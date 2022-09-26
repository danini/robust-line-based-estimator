// 
// \author Petr Hruby
// \date August 2022
// #include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "solver_2vp_22cl.cpp"

//PROTOTYPES

//samples projection of one vanishing point into two cameras whose relative pose is R,T
void sample_vp(const Eigen::Matrix3d R, //rotation between the cameras
		const Eigen::Vector3d T, //translation between the cameras
		Eigen::Vector3d &vp, //the vanishing point in the first camera
		Eigen::Vector3d &vq); //the vanishing point in the second camera

//samples one instance of the 2vp_22cl problem + its solution
void sample(Eigen::Vector3d * vps, //array of 2 sampled vanishing points in the first camera
	Eigen::Vector3d * vqs, //array of 2 sampled vanishing points in the second camera
	Eigen::Vector3d * ls,  //array of projections of 4 lines into 1st camera; first two lines are coplanar, second two lines are coplanar
	Eigen::Vector3d * ms,  //array of projections of 4 lines into 2nd camera; first two lines are coplanar, second two lines are coplanar
	Eigen::Matrix3d &R, //rotation between the cameras
	Eigen::Vector3d &T); //translation between the cameras

//solves one instance of the 2vp_22cl problem
int solver_wrapper_2vp_22cl(Eigen::Vector3d * vps, //array of 2 vanishing points in the first camera
				Eigen::Vector3d * vqs, //array of 2 vanishing points in the second camera
				Eigen::Vector3d * ls, //array of projections of 4 lines into 1st camera; first two lines are coplanar, second two lines are coplanar
				Eigen::Vector3d * ms, //array of projections of 4 lines into 2nd camera; first two lines are coplanar, second two lines are coplanar
				Eigen::Matrix3d * Rs, //rotations consistent with the input
				Eigen::Vector3d * ts); //translations consistent with the input

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


void sample(Eigen::Vector3d * vps, Eigen::Vector3d * vqs, Eigen::Vector3d * ls, Eigen::Vector3d * ms, Eigen::Matrix3d &R, Eigen::Vector3d &T)
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
	
	//GENERATE THE VANISHING POINTS
	Eigen::Vector3d vp1;
	Eigen::Vector3d vq1;
	sample_vp(R, T, vp1, vq1);
	vps[0] = vp1;
	vqs[0] = vq1;
	
	Eigen::Vector3d vp2;
	Eigen::Vector3d vq2;
	sample_vp(R, T, vp2, vq2);
	vps[1] = vp2;
	vqs[1] = vq2;
	
	//generate 2 pairs of coplanar lines
	//3 points -> 1st pair; another 3 points -> 2nd pair
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
	
	//project the points into both cameras
	Eigen::Vector3d p1 = X1/X1(2);
	Eigen::Vector3d p2 = X2/X2(2);
	Eigen::Vector3d p3 = X3/X3(2);
	Eigen::Vector3d p4 = X4/X4(2);
	Eigen::Vector3d p5 = X5/X5(2);
	Eigen::Vector3d p6 = X6/X6(2);
	
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
	
	//find the projections of the lines from the projections of the points
	Eigen::Vector3d l1 = p1.cross(p2);
	Eigen::Vector3d l2 = p1.cross(p3);
	
	Eigen::Vector3d l3 = p4.cross(p5);
	Eigen::Vector3d l4 = p4.cross(p6);
	
	Eigen::Vector3d m1 = q1.cross(q2);
	Eigen::Vector3d m2 = q1.cross(q3);
	
	Eigen::Vector3d m3 = q4.cross(q5);
	Eigen::Vector3d m4 = q4.cross(q6);
	
	ls[0] = l1;
	ls[1] = l2;
	ls[2] = l3;
	ls[3] = l4;
	
	ms[0] = m1;
	ms[1] = m2;
	ms[2] = m3;
	ms[3] = m4;
}

int solver_wrapper_2vp_22cl(Eigen::Vector3d * vps, Eigen::Vector3d * vqs, Eigen::Vector3d * ls, Eigen::Vector3d * ms, Eigen::Matrix3d * Rs, Eigen::Vector3d * ts)
{
	//find the rotation from the vps
	//normalize the vps
	Eigen::Vector3d vp1 = vps[0];
	Eigen::Vector3d vp2 = vps[1];
	Eigen::Vector3d vq1 = vqs[0];
	Eigen::Vector3d vq2 = vqs[1];
	vp1 = vp1/vp1.norm();
	vq1 = vq1/vq1.norm();
	vp2 = vp2/vp2.norm();
	vq2 = vq2/vq2.norm();

	//solve for R
	Eigen::Matrix3d R[4];
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
	R[0] = R1;
	R[1] = R2;
	R[2] = R3;
	R[3] = R4;
	
	//the equations:
	//det(M1) = 0, det(M2) = 0
	//data: l1,l2,l3,l4; m1,m2,m3,m4; rotation (columnwise): 33 parameters
	Eigen::VectorXd data = Eigen::VectorXd(33);
	data.block<3,1>(0,0) = ls[0];
	data.block<3,1>(3,0) = ls[1];
	data.block<3,1>(6,0) = ls[2];
	data.block<3,1>(9,0) = ls[3];
	
	data.block<3,1>(12,0) = ms[0];
	data.block<3,1>(15,0) = ms[1];
	data.block<3,1>(18,0) = ms[2];
	data.block<3,1>(21,0) = ms[3];
	
	int num_sols = 0;
	//for every rotation find the consistent translations
	for(int j=0;j<4;++j)
	{
		data.block<3,1>(24,0) = R[j].col(0);
		data.block<3,1>(27,0) = R[j].col(1);
		data.block<3,1>(30,0) = R[j].col(2);
	
		MatrixXcd t_sol = solver_2vp_22cl(data);
		for(int k=0;k<9;++k)
		{
			Eigen::Vector3d t;
			t << t_sol(0,k).real(), t_sol(1,k).real(), 1;
			t = t/t.norm();
			
			Rs[num_sols] = R[j];
			ts[num_sols] = t;
			++num_sols;
		}
	}
	return num_sols;
}
