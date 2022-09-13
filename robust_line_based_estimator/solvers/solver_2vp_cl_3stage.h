// 
// \author Petr Hruby
// \date July 2022
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

//compute a plane normal from a rotation matrix, a vanishing point, and a line coplanar with the vanishing point (the line may not contain the vanishing point)
int stage_2_normal_R_vp_l(const Eigen::Matrix3d R, //rotation matrix
						const Eigen::Vector3d vp, //vanishing point in 1st view
						const Eigen::Vector3d l, //line in 1st view
						const Eigen::Vector3d m, //line in 2nd view
						Eigen::Vector3d &n); //a normal of a plane that contains the vp and the line

//compute a translation from a rotation matrix, plane normal, and 3 lines coplanar with each other + with the vanishing point (at most 2 lines may contain the vanishing point)
int stage_3_translation_3l_R_n(const Eigen::Vector3d * l, //array of 3 lines in 1st view
							const Eigen::Vector3d * m, //array of the corresponding lines in 2st view
							const Eigen::Matrix3d R, //rotation matrix
							const Eigen::Vector3d n, //normal of the plane where lines l lie (precomputed in stage 2)
							Eigen::Vector3d &t); //a translation vector

//sample a vanishing point consistent with the rotation R and translation T
void sample_vp(const Eigen::Matrix3d R, //rotation matrix
				const Eigen::Vector3d T, //translation vector
				Eigen::Vector3d &vp, //vanishing point in 1st view
				Eigen::Vector3d &vq); //vanishing point in 2nd view

//sample: R, T, n, H, vp1, vq1, vp2, vq2, 3 lines
void sample(Eigen::Matrix3d &R, //rotation matrix
			Eigen::Vector3d &T, //translation vector
			Eigen::Vector3d &n, //normal of the plane
			Eigen::Matrix3d &H, //homography, H=R-T*n^T
			Eigen::Vector3d &vp1, //1st vanishing point in 1st view (coplanar with the plane related by H)
			Eigen::Vector3d &vq1, //1st vanishing point in 2nd view (coplanar with the plane related by H)
			Eigen::Vector3d &vp2, //2nd vanishing point in 1st view
			Eigen::Vector3d &vq2, //2nd vanishing point in 2nd view
			Eigen::Vector3d * ls, //array of 3 lines in 1st view; the lines are coplanar with the plane related by H
			Eigen::Vector3d * ms); //array of 3 lines in 2nd view; the lines are coplanar with the plane related by H

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

int stage_2_normal_R_vp_l(const Eigen::Matrix3d R, const Eigen::Vector3d vp, const Eigen::Vector3d l, const Eigen::Vector3d m, Eigen::Vector3d &n)
{
	//the vp constrains the normal, as vp^T*n = 0 => n(1)=a*n(0)+b
	const double a = -vp(0)/vp(1);
	const double b = -vp(2)/vp(1);
	
	//
	double l0 = l(0);
	double l1 = l(1);
	double l2 = l(2);
	double m0 = m(0);
	double m1 = m(1);
	double m2 = m(2);
	
	//if the normal is correct, the homography constraint imposes 1 LI constraint on the translation, otherwise, it imposes 2 LI constraints
	//take 1 line, use the hidden variable approach to eliminate translation: the constraint on n is the determinant of 1st+last cols (TODO why does this work?)
	double c1 = a*l2*l2*m0*m0*R(0,0) - a*l0*l2*m0*m0*R(0,2) + a*l2*l2*m0*m1*R(1,0) - a*l0*l2*m0*m1*R(1,2) + a*l2*l2*m0*m2*R(2,0) - a*l0*l2*m0*m2*R(2,2) - l2*l2*m0*m0*R(0,1) + l1*l2*m0*m0*R(0,2) - l2*l2*m0*m1*R(1,1) + l1*l2*m0*m1*R(1,2) - l2*l2*m0*m2*R(2,1) + l1*l2*m0*m2*R(2,2);
	double c0 = b*l2*l2*m0*m0*R(0,0) - b*l0*l2*m0*m0*R(0,2) + b*l2*l2*m0*m1*R(1,0) - b*l0*l2*m0*m1*R(1,2) + b*l2*l2*m0*m2*R(2,0) - b*l0*l2*m0*m2*R(2,2) - l1*l2*m0*m0*R(0,0) + l0*l2*m0*m0*R(0,1) - l1*l2*m0*m1*R(1,0) + l0*l2*m0*m1*R(1,1) - l1*l2*m0*m2*R(2,0) + l0*l2*m0*m2*R(2,1);
	double n0 = -c0/c1;
	
	//compose the normal vector
	n(0) = n0;
	n(1) = a*n0+b;
	n(2) = 1;
	
	return 1;
}

int stage_3_translation_3l_R_n(const Eigen::Vector3d * l, const Eigen::Vector3d * m, const Eigen::Matrix3d R, const Eigen::Vector3d n, Eigen::Vector3d &t)
{
	Eigen::Matrix<double,6,4> MM;
	
	//the homography constraint becomes linear after both the rotation and the normal are known 
	MM(0,0) = n(0)*l[0](2)*m[0](0) - l[0](0)*m[0](0);
	MM(0,1) = n(0)*l[0](2)*m[0](1) - l[0](0)*m[0](1); //1st 3 cols in the matrix are always LD, LI for nonsolutions is given by the last col
	MM(0,2) = n(0)*l[0](2)*m[0](2) - l[0](0)*m[0](2);
	MM(0,3) = -l[0](2)*m[0](0)*R(0,0) + l[0](0)*m[0](0)*R(0,2) - l[0](2)*m[0](1)*R(1,0) + l[0](0)*m[0](1)*R(1,2) - l[0](2)*m[0](2)*R(2,0) + l[0](0)*m[0](2)*R(2,2);
	
	MM(1,0) = n(1)*l[0](2)*m[0](0) - l[0](1)*m[0](0);
	MM(1,1) = n(1)*l[0](2)*m[0](1) - l[0](1)*m[0](1);
	MM(1,2) = n(1)*l[0](2)*m[0](2) - l[0](1)*m[0](2);
	MM(1,3) = -l[0](2)*m[0](0)*R(0,1) + l[0](1)*m[0](0)*R(0,2) - l[0](2)*m[0](1)*R(1,1) + l[0](1)*m[0](1)*R(1,2) - l[0](2)*m[0](2)*R(2,1) + l[0](1)*m[0](2)*R(2,2);
	
	MM(2,0) = n(0)*l[1](2)*m[1](0) - l[1](0)*m[1](0);
	MM(2,1) = n(0)*l[1](2)*m[1](1) - l[1](0)*m[1](1);
	MM(2,2) = n(0)*l[1](2)*m[1](2) - l[1](0)*m[1](2);
	MM(2,3) = -l[1](2)*m[1](0)*R(0,0) + l[1](0)*m[1](0)*R(0,2) - l[1](2)*m[1](1)*R(1,0) + l[1](0)*m[1](1)*R(1,2) - l[1](2)*m[1](2)*R(2,0) + l[1](0)*m[1](2)*R(2,2);
	
	MM(3,0) = n(1)*l[1](2)*m[1](0) - l[1](1)*m[1](0);
	MM(3,1) = n(1)*l[1](2)*m[1](1) - l[1](1)*m[1](1);
	MM(3,2) = n(1)*l[1](2)*m[1](2) - l[1](1)*m[1](2);
	MM(3,3) = -l[1](2)*m[1](0)*R(0,1) + l[1](1)*m[1](0)*R(0,2) - l[1](2)*m[1](1)*R(1,1) + l[1](1)*m[1](1)*R(1,2) - l[1](2)*m[1](2)*R(2,1) + l[1](1)*m[1](2)*R(2,2);
	
	MM(4,0) = n(0)*l[2](2)*m[2](0) - l[2](0)*m[2](0);
	MM(4,1) = n(0)*l[2](2)*m[2](1) - l[2](0)*m[2](1);
	MM(4,2) = n(0)*l[2](2)*m[2](2) - l[2](0)*m[2](2);
	MM(4,3) = -l[2](2)*m[2](0)*R(0,0) + l[2](0)*m[2](0)*R(0,2) - l[2](2)*m[2](1)*R(1,0) + l[2](0)*m[2](1)*R(1,2) - l[2](2)*m[2](2)*R(2,0) + l[2](0)*m[2](2)*R(2,2);
	
	MM(5,0) = n(1)*l[2](2)*m[2](0) - l[2](1)*m[2](0);
	MM(5,1) = n(1)*l[2](2)*m[2](1) - l[2](1)*m[2](1);
	MM(5,2) = n(1)*l[2](2)*m[2](2) - l[2](1)*m[2](2);
	MM(5,3) = - l[2](2)*m[2](0)*R(0,1) + l[2](1)*m[2](0)*R(0,2) - l[2](2)*m[2](1)*R(1,1) + l[2](1)*m[2](1)*R(1,2) - l[2](2)*m[2](2)*R(2,1) + l[2](1)*m[2](2)*R(2,2);
	
	
	
	Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(MM, Eigen::ComputeFullU | Eigen::ComputeFullV);	
	Eigen::MatrixXd V = svd.matrixV();
	//std::cout << V << "\n\n";
	Eigen::Vector4d h = V.col(3);
	h = h/h(3);
	t = h.block<3,1>(0,0);
		
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

void sample(Eigen::Matrix3d &R, Eigen::Vector3d &T, Eigen::Vector3d &n, Eigen::Matrix3d &H, Eigen::Vector3d &vp1, Eigen::Vector3d &vq1, Eigen::Vector3d &vp2, Eigen::Vector3d &vq2, Eigen::Vector3d * ls, Eigen::Vector3d * ms)
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
	
	//GENERATE A HOMOGRAPHY
	//generate 4 points on a plane
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
	
	//generate 4th point on the plane defined by the 3 points
	double a4 = norm_sampler(eng);
	double b4 = norm_sampler(eng);
	Eigen::Vector3d X4 = X1 + a4*(X2-X1) + b4*(X3-X1);
	
	//project the points into the cameras
	Eigen::Vector3d p1 = X1/X1(2);
	Eigen::Vector3d p2 = X2/X2(2);
	Eigen::Vector3d p3 = X3/X3(2);
	Eigen::Vector3d p4 = X4/X4(2);
	
	Eigen::Vector3d q1 = R*X1+T;
	q1 = q1/q1(2);
	Eigen::Vector3d q2 = R*X2+T;
	q2 = q2/q2(2);
	Eigen::Vector3d q3 = R*X3+T;
	q3 = q3/q3(2);
	Eigen::Vector3d q4 = R*X4+T;
	q4 = q4/q4(2);
	
	//compute the homography from the points
	Eigen::Matrix<double, 8, 9> M = Eigen::Matrix<double, 8, 9>::Zero(8,9);
	M.block<1,3>(0,0) = p1.transpose();
	M.block<1,3>(0,6) = -q1(0)*p1.transpose();
	M.block<1,3>(1,3) = p1.transpose();
	M.block<1,3>(1,6) = -q1(1)*p1.transpose();
	
	M.block<1,3>(2,0) = p2.transpose();
	M.block<1,3>(2,6) = -q2(0)*p2.transpose();
	M.block<1,3>(3,3) = p2.transpose();
	M.block<1,3>(3,6) = -q2(1)*p2.transpose();
	
	M.block<1,3>(4,0) = p3.transpose();
	M.block<1,3>(4,6) = -q3(0)*p3.transpose();
	M.block<1,3>(5,3) = p3.transpose();
	M.block<1,3>(5,6) = -q3(1)*p3.transpose();
	
	M.block<1,3>(6,0) = p4.transpose();
	M.block<1,3>(6,6) = -q4(0)*p4.transpose();
	M.block<1,3>(7,3) = p4.transpose();
	M.block<1,3>(7,6) = -q4(1)*p4.transpose();
	
	Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> USV(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXd V = USV.matrixV();
	Eigen::VectorXd h = V.col(8);
	H.block<1,3>(0,0) = h.block<3,1>(0,0).transpose();
	H.block<1,3>(1,0) = h.block<3,1>(3,0).transpose();
	H.block<1,3>(2,0) = h.block<3,1>(6,0).transpose();
	
	//DECOMPOSE THE HOMOGRAPHY TO GET THE NORMAL
	//alpha*R(:,0) - beta(0)*T = H(:,0)
	//[R(:,0) -T] * [alpha beta(0)] = H(:,0) -> solve for alpha, beta(0) using least squares; [beta(0), beta(1), beta(2)]^T ~ n
	Eigen::Matrix<double, 3,2> AA0;
	AA0.block<3,1>(0,0) = R.block<3,1>(0,0);
	AA0.block<3,1>(0,1) = -T;
	Eigen::Vector2d ab_sol0 = AA0.colPivHouseholderQr().solve(H.block<3,1>(0,0));
	
	Eigen::Matrix<double, 3,2> AA1;
	AA1.block<3,1>(0,0) = R.block<3,1>(0,1);
	AA1.block<3,1>(0,1) = -T;
	Eigen::Vector2d ab_sol1 = AA1.colPivHouseholderQr().solve(H.block<3,1>(0,1));
	
	Eigen::Matrix<double, 3,2> AA2;
	AA2.block<3,1>(0,0) = R.block<3,1>(0,2);
	AA2.block<3,1>(0,1) = -T;
	Eigen::Vector2d ab_sol2 = AA2.colPivHouseholderQr().solve(H.block<3,1>(0,2));
	
	Eigen::Vector3d nn;
	nn(0) = ab_sol0(1)/ab_sol0(0);
	nn(1) = ab_sol1(1)/ab_sol1(0);
	nn(2) = ab_sol2(1)/ab_sol2(0);
	
	//SAMPLE 1VP + 2 LINES LYING ON THE PLANE TRANSFORMED BY THE HOMOGRAPHY	
	Eigen::Vector3d dir = X2-X1;
	Eigen::Vector3d X5 = X3+dir;
	Eigen::Vector3d X6 = X4+dir;
	
	double a7 = norm_sampler(eng);
	double b7 = norm_sampler(eng);
	Eigen::Vector3d X7 = X1 + a7*(X2-X1) + b7*(X3-X1);
	double a8 = norm_sampler(eng);
	double b8 = norm_sampler(eng);
	Eigen::Vector3d X8 = X1 + a8*(X2-X1) + b8*(X3-X1);
	
	double a9 = norm_sampler(eng);
	double b9 = norm_sampler(eng);
	Eigen::Vector3d X9 = X1 + a9*(X2-X1) + b9*(X3-X1);
	double a10 = norm_sampler(eng);
	double b10 = norm_sampler(eng);
	Eigen::Vector3d X10 = X1 + a10*(X2-X1) + b10*(X3-X1);
	
	double a11 = norm_sampler(eng);
	double b11 = norm_sampler(eng);
	Eigen::Vector3d X11 = X1 + a11*(X2-X1) + b11*(X3-X1);
	double a12 = norm_sampler(eng);
	double b12 = norm_sampler(eng);
	Eigen::Vector3d X12 = X1 + a12*(X2-X1) + b12*(X3-X1);
	
	Eigen::Vector3d X13 = X7+dir;
	
	//project them to the first view
	Eigen::Vector3d p5 = X5/X5(2);
	Eigen::Vector3d p6 = X6/X6(2);
	Eigen::Vector3d p7 = X7/X7(2);
	Eigen::Vector3d p8 = X8/X8(2);
	Eigen::Vector3d p9 = X9/X9(2);
	Eigen::Vector3d p10 = X10/X10(2);
	Eigen::Vector3d p11 = X11/X11(2);
	Eigen::Vector3d p12 = X12/X12(2);
	Eigen::Vector3d p13 = X13/X13(2);
	
	//find the projections of the lines
	Eigen::Vector3d l3 = p3.cross(p5);
	Eigen::Vector3d l4 = p4.cross(p6);
	Eigen::Vector3d l5 = p7.cross(p9);
	Eigen::Vector3d l6 = p8.cross(p10);
	Eigen::Vector3d l7 = p11.cross(p12);
	Eigen::Vector3d l8 = p7.cross(p13);
	
	//also project them to the second view + find the lines
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
	Eigen::Vector3d q10 = R*X10+T;
	q10 = q10/q10(2);
	Eigen::Vector3d q11 = R*X11+T;
	q11 = q11/q11(2);
	Eigen::Vector3d q12 = R*X12+T;
	q12 = q12/q12(2);
	Eigen::Vector3d q13 = R*X13+T;
	q13 = q13/q13(2);
	
	Eigen::Vector3d m3 = q3.cross(q5);
	Eigen::Vector3d m4 = q4.cross(q6);
	Eigen::Vector3d m5 = q7.cross(q9);
	Eigen::Vector3d m6 = q8.cross(q10);
	Eigen::Vector3d m7 = q11.cross(q12);
	Eigen::Vector3d m8 = q7.cross(q13);
	
	ls[0] = l5;
	ls[1] = l6;
	ls[2] = l7;
	
	ms[0] = m5;
	ms[1] = m6;
	ms[2] = m7;
	
	//find the vanishing point consistent with the plane
	vp1 = l3.cross(l4);
	vq1 = m3.cross(m4);
	
	//sample the second vanishing point
	sample_vp(R, T, vp2, vq2);

	//normalize the values	
	n = nn/nn(2);
	T = T*nn(2);
	Eigen::Vector3d tc = T*nn(2);
	double n0 = n(0);
	double t0 = tc(0);
	double t1 = tc(1);
	double t2 = tc(2);

}

