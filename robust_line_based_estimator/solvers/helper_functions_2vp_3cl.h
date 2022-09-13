// 
// \author Petr Hruby
// \date June 2022
// #include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "solver_cpp_2vp_3cl_hidden.cpp"

//PROTOTYPES
//compute a rotation from two vanishing points
int stage_1_solver_rotation_2vp(Eigen::Vector3d vp1, //1st vanishing point in 1st view
								Eigen::Vector3d vq1, //1st vanishing point in 2nd view
								Eigen::Vector3d vp2, //2nd vanishing point in 1st view
								Eigen::Vector3d vq2, //2nd vanishing point in 2nd view
								Eigen::Matrix3d * Rs); //set of rotations consistent with the vanishing points (4 rotations)

//compute a translation from 3 coplanar lines + known rotation
//returns a complex 2*6 matrix whose rows are the translations consistent with the input; x- and y- coordinates of the translation are computed, z-coordinate of the vector is assumed to be 1
Eigen::MatrixXcd stage_2_wrapper_translation_3cl(const Eigen::Vector3d l1, //1st line in 1st view
									const Eigen::Vector3d m1, //1st line in 2nd view
									const Eigen::Vector3d l2, //2nd line in 1st view
									const Eigen::Vector3d m2, //2nd line in 2nd view
									const Eigen::Vector3d l3, //3rd line in 1st view
									const Eigen::Vector3d m3, //3rd line in 2nd view
									const Eigen::Matrix3d R); //rotation matrix

//sample 2 vps, 3 coplanar lines + a relative pose
void sample(Eigen::Vector3d * vps, //array of 2 vanishing points in the 1st view
			Eigen::Vector3d * vqs, //array of 2 vanishing points in the 1st view
			Eigen::Vector3d * ls, //array of 2 vanishing points in the 1st view
			Eigen::Vector3d * ms, //array of 2 vanishing points in the 1st view
			Eigen::Matrix3d &R, //rotation matrix
			Eigen::Vector3d &T); //translation vector

//sample a vanishing point consistent with the rotation R and translation T
void sample_vp(const Eigen::Matrix3d R, //rotation matrix
				const Eigen::Vector3d T, //translation vector
				Eigen::Vector3d &vp, //vanishing point in 1st view
				Eigen::Vector3d &vq); //vanishing point in 2nd view

//FUNCTIONS
int stage_1_solver_rotation_2vp(Eigen::Vector3d vp1, Eigen::Vector3d vq1, Eigen::Vector3d vp2, Eigen::Vector3d vq2, Eigen::Matrix3d * Rs)
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

Eigen::MatrixXcd stage_2_wrapper_translation_3cl(const Eigen::Vector3d l1, //1st line in 1st view
										const Eigen::Vector3d m1, //1st line in 2nd view
										const Eigen::Vector3d l2, //2nd line in 1st view
										const Eigen::Vector3d m2, //2nd line in 2nd view
										const Eigen::Vector3d l3, //3rd line in 1st view
										const Eigen::Vector3d m3, //3rd line in 2nd view
										const Eigen::Matrix3d R) //rotation matrix
{
	//fill the data vector
	Eigen::VectorXd data = Eigen::VectorXd(27);
	data.block<3,1>(0,0) = R.row(0).transpose();
	data.block<3,1>(3,0) = R.row(1).transpose();
	data.block<3,1>(6,0) = R.row(2).transpose();
	data.block<3,1>(9,0) = l1;
	data.block<3,1>(12,0) = m1;
	data.block<3,1>(15,0) = l2;
	data.block<3,1>(18,0) = m2;
	data.block<3,1>(21,0) = l3;
	data.block<3,1>(24,0) = m3;
	
	//run the solver
	return solver_cpp_2vp_3cl_hidden(data);
}

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
	
	//GENERATE 2 VANISHING POINTS
	Eigen::Vector3d vp1;
	Eigen::Vector3d vq1;
	Eigen::Vector3d vp2;
	Eigen::Vector3d vq2;
	sample_vp(R, T, vp1, vq1);
	sample_vp(R, T, vp2, vq2);
	//std::cout << vp1 << "\n\n";
	vps[0] = vp1;
	vps[1] = vp2;
	vqs[0] = vq1;
	vqs[1] = vq2;
	
	//GENERATE THE LINES
	//generate 3 points
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
	
	//project the points into the cameras
	Eigen::Vector3d p1 = X1/X1(2);
	Eigen::Vector3d p2 = X2/X2(2);
	Eigen::Vector3d p3 = X3/X3(2);
	
	Eigen::Vector3d q1 = R*X1+T;
	q1 = q1/q1(2);
	Eigen::Vector3d q2 = R*X2+T;
	q2 = q2/q2(2);
	Eigen::Vector3d q3 = R*X3+T;
	q3 = q3/q3(2);
	
	//find the projections of 3 lines from the projections of each pair of the points
	Eigen::Vector3d l1 = p1.cross(p2);
	Eigen::Vector3d l2 = p1.cross(p3);
	Eigen::Vector3d l3 = p2.cross(p3);
	
	Eigen::Vector3d m1 = q1.cross(q2);
	Eigen::Vector3d m2 = q1.cross(q3);
	Eigen::Vector3d m3 = q2.cross(q3);
	
	ls[0] = l1;
	ls[1] = l2;
	ls[2] = l3;
	
	ms[0] = m1;
	ms[1] = m2;
	ms[2] = m3;
}

