// 
// \author Petr Hruby
// \date June 2022
// #include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES
//compute a rotation from two vanishing points
int stage_1_solver_rotation_2vp(Eigen::Vector3d vp1, //1st vanishing point in 1st view
								Eigen::Vector3d vq1, //1st vanishing point in 2nd view
								Eigen::Vector3d vp2, //2nd vanishing point in 1st view
								Eigen::Vector3d vq2, //2nd vanishing point in 2nd view
								Eigen::Matrix3d * Rs); //set of rotations consistent with the vanishing points (4 rotations)

//compute a translation from 2 points + known rotation
int stage_2_solver_translation_2pt(const Eigen::Vector3d p1, //1st point in 1st view
									const Eigen::Vector3d q1, //1st point in 2nd view
									const Eigen::Vector3d p2, //2nd point in 1st view
									const Eigen::Vector3d q2, //2nd point in 2nd view
									const Eigen::Matrix3d R, //rotation matrix
									Eigen::Vector3d &t); //translation vector

void sample(Eigen::Vector3d * vps, //array of 2 vanishing points in 1st view
			Eigen::Vector3d * vqs, //array of 2 vanishing points in 2nd view
			Eigen::Vector3d * pts, //array of 2 points in 1st view
			Eigen::Vector3d * qts, //array of 2 points in 2nd view
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

int stage_2_solver_translation_2pt(const Eigen::Vector3d p1, const Eigen::Vector3d q1, const Eigen::Vector3d p2, const Eigen::Vector3d q2, const Eigen::Matrix3d R, Eigen::Vector3d &t)
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

void sample(Eigen::Vector3d * vps, Eigen::Vector3d * vqs, Eigen::Vector3d * pts, Eigen::Vector3d * qts, Eigen::Matrix3d &R, Eigen::Vector3d &T)
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
	
	//GENERATE 2 FINITE POINTS (mean [0 0 5], std 1)
	double x1 = norm_sampler(eng);
	double y1 = norm_sampler(eng);
	double z1 = norm_sampler(eng);
	Eigen::Vector3d X1(x1, y1, z1+5);
	
	double x2 = norm_sampler(eng);
	double y2 = norm_sampler(eng);
	double z2 = norm_sampler(eng);
	Eigen::Vector3d X2(x2, y2, z2+5);
	
	//compute the projections of the points
	Eigen::Vector3d p1 = X1/X1(0);
	Eigen::Vector3d q1 = R*X1+T;
	q1 = q1/q1(0);
	pts[0] = p1;
	qts[0] = q1;
	
	Eigen::Vector3d p2 = X2/X2(0);
	Eigen::Vector3d q2 = R*X2+T;
	q2 = q2/q2(0);
	pts[1] = p2;
	qts[1] = q2;
}

