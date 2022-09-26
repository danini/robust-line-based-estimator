// 
// \author Petr Hruby
// \date August 2022
// #include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES
//compute relative pose from 1 vanishing point and 3 points
int solver_wrapper_1vp_3pt(Eigen::Vector3d vp, //the vanishing point in 1st view
					Eigen::Vector3d vq, //the vanishing point in 2nd view
					Eigen::Vector3d * pts, //array of 3 points in the first view
					Eigen::Vector3d * qts, //array of 3 points in the second view
					Eigen::Matrix3d * Rs, //set of rotations consistent with the input data
					Eigen::Vector3d * Ts); //set of translation vectors consistent with the input data

//solve the polynomial system arising from the 1vp_3pt problem				
//returns a matrix of (possibly complex) rotation parameters consistent with the input
Eigen::MatrixXcd solver_1vp_3pt(Eigen::Vector3d * p, //rectified points in the first view
				Eigen::Vector3d * q); //rectified points in the second view


//sample one instance of the problem + its solution
void sample(Eigen::Vector3d * vps, //array of 1 vanishing point in 1st view
			Eigen::Vector3d * vqs, //array of 1 vanishing point in 2nd view
			Eigen::Vector3d * pts, //array of 3 points in 1st view
			Eigen::Vector3d * qts, //array of 3 points in 2nd view
			Eigen::Matrix3d &R, //rotation matrix
			Eigen::Vector3d &T); //translation vector

//sample a vanishing point consistent with the rotation R and translation T
void sample_vp(const Eigen::Matrix3d R, //rotation matrix
				const Eigen::Vector3d T, //translation vector
				Eigen::Vector3d &vp, //vanishing point in 1st view
				Eigen::Vector3d &vq); //vanishing point in 2nd view
				
//FUNCTIONS
Eigen::MatrixXcd solver_1vp_3pt(Eigen::Vector3d * p, Eigen::Vector3d * q)
{
	//build vectors mA, mB, mC, mD, mE, mF, mG, mH, mJ, such that:
		//matrix M(t) = t*[mA mD mG] + (1-t*t)*[mB mE mH] + (1+t*t)*[mC mF mJ]
		//and M(t)*T = 0
	Eigen::Vector3d mA;
	mA << -2*p[0](0)*q[0](1), -2*p[1](0)*q[1](1), -2*p[2](0)*q[2](1);
	Eigen::Vector3d mB;
	mB << -p[0](2)*q[0](1), -p[1](2)*q[1](1), -p[2](2)*q[2](1);
	Eigen::Vector3d mC;
	mC << p[0](1)*q[0](2), p[1](1)*q[1](2), p[2](1)*q[2](2);
	
	Eigen::Vector3d mD;
	mD << 2*p[0](2)*q[0](2) + 2*p[0](0)*q[0](0), 2*p[1](2)*q[1](2) + 2*p[1](0)*q[1](0), 2*p[2](2)*q[2](2) + 2*p[2](0)*q[2](0);
	Eigen::Vector3d mE;
	mE << -p[0](0)*q[0](2) + p[0](2)*q[0](0), -p[1](0)*q[1](2) + p[1](2)*q[1](0), -p[2](0)*q[2](2) + p[2](2)*q[2](0);
	//mF = [0 0 0]^T
	
	Eigen::Vector3d mG;
	mG << -2*p[0](2)*q[0](1), -2*p[1](2)*q[1](1), -2*p[2](2)*q[2](1);
	Eigen::Vector3d mH;
	mH << p[0](0)*q[0](1), p[1](0)*q[1](1), p[2](0)*q[2](1);
	Eigen::Vector3d mJ;
	mJ << -p[0](1)*q[0](0), -p[1](1)*q[1](0), -p[2](1)*q[2](0);
	
	Eigen::Matrix3d Mp;
	Mp.col(0) = mC;
	Mp.col(1) = Eigen::Vector3d::Zero();
	Mp.col(2) = mJ;
	
	Eigen::Matrix3d Mx;
	Mx.col(0) = mA;
	Mx.col(1) = mD;
	Mx.col(2) = mG;
	
	Eigen::Matrix3d Mm;
	Mm.col(0) = mB;
	Mm.col(1) = mE;
	Mm.col(2) = mH;
	
	//the determinant of det(t) of matrix M(t) is a polynomial of degree 6
	//find coefficients c0, c1, c2, c3, c4, c5, c6, such that det(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5 + c6*t^6		
	const double m01 = mA(0)*mD(1)*mG(2);
	const double m02 = mA(0)*mD(1)*mH(2) + mA(0)*mE(1)*mG(2) + mB(0)*mD(1)*mG(2);
	const double m03 = mA(0)*mD(1)*mJ(2) + mC(0)*mD(1)*mG(2);
	const double m04 = mA(0)*mE(1)*mH(2) + mB(0)*mD(1)*mH(2) + mB(0)*mE(1)*mG(2);
	const double m05 = mA(0)*mE(1)*mJ(2) + mB(0)*mD(1)*mJ(2) + mC(0)*mD(1)*mH(2) + mC(0)*mE(1)*mG(2);
	const double m06 = mC(0)*mD(1)*mJ(2);
	const double m07 = mB(0)*mE(1)*mH(2);
	const double m08 = mB(0)*mE(1)*mJ(2) + mC(0)*mE(1)*mH(2);
	const double m09 = mC(0)*mE(1)*mJ(2);
	
	const double m11 = -mA(0)*mD(2)*mG(1);
	const double m12 = -mA(0)*mD(2)*mH(1) - mA(0)*mE(2)*mG(1) - mB(0)*mD(2)*mG(1);
	const double m13 = -mA(0)*mD(2)*mJ(1) - mC(0)*mD(2)*mG(1);
	const double m14 = -mA(0)*mE(2)*mH(1) - mB(0)*mD(2)*mH(1) - mB(0)*mE(2)*mG(1);
	const double m15 = -mA(0)*mE(2)*mJ(1) - mB(0)*mD(2)*mJ(1) - mC(0)*mD(2)*mH(1) - mC(0)*mE(2)*mG(1);
	const double m16 = -mC(0)*mD(2)*mJ(1);
	const double m17 = -mB(0)*mE(2)*mH(1);
	const double m18 = -mB(0)*mE(2)*mJ(1) - mC(0)*mE(2)*mH(1);
	const double m19 = -mC(0)*mE(2)*mJ(1);
	
	const double m21 = -mA(1)*mD(0)*mG(2);
	const double m22 = -mA(1)*mD(0)*mH(2) - mA(1)*mE(0)*mG(2) - mB(1)*mD(0)*mG(2);
	const double m23 = -mA(1)*mD(0)*mJ(2) - mC(1)*mD(0)*mG(2);
	const double m24 = -mA(1)*mE(0)*mH(2) - mB(1)*mD(0)*mH(2) - mB(1)*mE(0)*mG(2);
	const double m25 = -mA(1)*mE(0)*mJ(2) - mB(1)*mD(0)*mJ(2) - mC(1)*mD(0)*mH(2) - mC(1)*mE(0)*mG(2);
	const double m26 = -mC(1)*mD(0)*mJ(2);
	const double m27 = -mB(1)*mE(0)*mH(2);
	const double m28 = -mB(1)*mE(0)*mJ(2) - mC(1)*mE(0)*mH(2);
	const double m29 = -mC(1)*mE(0)*mJ(2);
	
	const double m31 = mA(1)*mD(2)*mG(0);
	const double m32 = mA(1)*mD(2)*mH(0) + mA(1)*mE(2)*mG(0) + mB(1)*mD(2)*mG(0);
	const double m33 = mA(1)*mD(2)*mJ(0) + mC(1)*mD(2)*mG(0);
	const double m34 = mA(1)*mE(2)*mH(0) + mB(1)*mD(2)*mH(0) + mB(1)*mE(2)*mG(0);
	const double m35 = mA(1)*mE(2)*mJ(0) + mB(1)*mD(2)*mJ(0) + mC(1)*mD(2)*mH(0) + mC(1)*mE(2)*mG(0);
	const double m36 = mC(1)*mD(2)*mJ(0);
	const double m37 = mB(1)*mE(2)*mH(0);
	const double m38 = mB(1)*mE(2)*mJ(0) + mC(1)*mE(2)*mH(0);
	const double m39 = mC(1)*mE(2)*mJ(0);
	
	const double m41 = mA(2)*mD(0)*mG(1);
	const double m42 = mA(2)*mD(0)*mH(1) + mA(2)*mE(0)*mG(1) + mB(2)*mD(0)*mG(1);
	const double m43 = mA(2)*mD(0)*mJ(1) + mC(2)*mD(0)*mG(1);
	const double m44 = mA(2)*mE(0)*mH(1) + mB(2)*mD(0)*mH(1) + mB(2)*mE(0)*mG(1);
	const double m45 = mA(2)*mE(0)*mJ(1) + mB(2)*mD(0)*mJ(1) + mC(2)*mD(0)*mH(1) + mC(2)*mE(0)*mG(1);
	const double m46 = mC(2)*mD(0)*mJ(1);
	const double m47 = mB(2)*mE(0)*mH(1);
	const double m48 = mB(2)*mE(0)*mJ(1) + mC(2)*mE(0)*mH(1);
	const double m49 = mC(2)*mE(0)*mJ(1);
	
	const double m51 = -mA(2)*mD(1)*mG(0);
	const double m52 = -mA(2)*mD(1)*mH(0) - mA(2)*mE(1)*mG(0) - mB(2)*mD(1)*mG(0);
	const double m53 = -mA(2)*mD(1)*mJ(0) - mC(2)*mD(1)*mG(0);
	const double m54 = -mA(2)*mE(1)*mH(0) - mB(2)*mD(1)*mH(0) - mB(2)*mE(1)*mG(0);
	const double m55 = -mA(2)*mE(1)*mJ(0) - mB(2)*mD(1)*mJ(0) - mC(2)*mD(1)*mH(0) - mC(2)*mE(1)*mG(0);
	const double m56 = -mC(2)*mD(1)*mJ(0);
	const double m57 = -mB(2)*mE(1)*mH(0);
	const double m58 = -mB(2)*mE(1)*mJ(0) - mC(2)*mE(1)*mH(0);
	const double m59 = -mC(2)*mE(1)*mJ(0);
	
	const double c0 = m07 + m08 + m09 + m17 + m18 + m19 + m27 + m28 + m29 + m37 + m38 + m39 + m47 + m48 + m49 + m57 + m58 + m59;
	const double c1 = m04 + m05 + m06 + m14 + m15 + m16 + m24 + m25 + m26 + m34 + m35 + m36 + m44 + m45 + m46 + m54 + m55 + m56;
	const double c2 = m02+m03-3*m07-m08+m09 + m12+m13-3*m17-m18+m19 + m22+m23-3*m27-m28+m29 + m32+m33-3*m37-m38+m39 + m42+m43-3*m47-m48+m49 + m52+m53-3*m57-m58+m59;
	const double c3 = m01-2*m04+2*m06 + m11-2*m14+2*m16 + m21-2*m24+2*m26 + m31-2*m34+2*m36 + m41-2*m44+2*m46 + m51-2*m54+2*m56;
	const double c4 = -m02+m03+3*m07-m08-m09 -m12+m13+3*m17-m18-m19 -m22+m23+3*m27-m28-m29 -m32+m33+3*m37-m38-m39 -m42+m43+3*m47-m48-m49 -m52+m53+3*m57-m58-m59;
	const double c5 = m04 - m05 + m06 + m14 - m15 + m16 + m24 - m25 + m26 + m34 - m35 + m36 + m44 - m45 + m46 + m54 - m55 + m56;
	const double c6 = -m07 + m08 - m09 - m17 + m18 - m19 - m27 + m28 - m29 - m37 + m38 - m39 - m47 + m48 - m49 - m57 + m58 - m59;
	
	//solve equation det(t)=0 by a companion matrix approach
	Eigen::Matrix<double,6,6> C = Eigen::Matrix<double,6,6>::Zero();
	C(1,0) = 1;
	C(2,1) = 1;
	C(3,2) = 1;
	C(4,3) = 1;
	C(5,4) = 1;
	
	C(0,5) = -c0/c6;
	C(1,5) = -c1/c6;
	C(2,5) = -c2/c6;
	C(3,5) = -c3/c6;
	C(4,5) = -c4/c6;
	C(5,5) = -c5/c6;
	
	return C.eigenvalues();
}

int solver_wrapper_1vp_3pt(Eigen::Vector3d vp, Eigen::Vector3d vq, Eigen::Vector3d * pts, Eigen::Vector3d * qts, Eigen::Matrix3d * Rs, Eigen::Vector3d * Ts)
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
	}
	
	//solve the problem for the original direction
	const Eigen::MatrixXcd evs = solver_1vp_3pt(p, q);
	
	//find the real solutions, build the relative poses and unrectify them
	int num_real_sols = 0;
	for(int i=0;i<6;++i)
	{
		if(evs(i).imag() < 1e-10 && evs(i).imag() > -1e-10)
		{
			//real solution
			const double x = evs(i).real();
			
			//build the rotation matrix of the rectified problem
			Eigen::Matrix3d RR;
			RR << (1-x*x)/(1+x*x), 0, -2*x/(1+x*x), 0, 1, 0, 2*x/(1+x*x), 0, (1-x*x)/(1+x*x);
			
			//find the translation of the rectified problem
			Eigen::Matrix3d M;
			M(0,0) = p[0](1)*(1+x*x)*q[0](2) - (2*p[0](0)*x + p[0](2)*(1-x*x))*q[0](1);
			M(0,1) = -(p[0](0)*(1-x*x)-2*p[0](2)*x)*q[0](2) + (2*p[0](0)*x + p[0](2)*(1-x*x))*q[0](0);
			M(0,2) = (p[0](0)*(1-x*x)-2*p[0](2)*x)*q[0](1) - p[0](1)*(1+x*x)*q[0](0);
			
			M(1,0) = p[1](1)*(1+x*x)*q[1](2) - (2*p[1](0)*x + p[1](2)*(1-x*x))*q[1](1);
			M(1,1) = -(p[1](0)*(1-x*x)-2*p[1](2)*x)*q[1](2) + (2*p[1](0)*x + p[1](2)*(1-x*x))*q[1](0);
			M(1,2) = (p[1](0)*(1-x*x)-2*p[1](2)*x)*q[1](1) - p[1](1)*(1+x*x)*q[1](0);
			
			M(2,0) = p[2](1)*(1+x*x)*q[2](2) - (2*p[2](0)*x + p[2](2)*(1-x*x))*q[2](1);
			M(2,1) = -(p[2](0)*(1-x*x)-2*p[2](2)*x)*q[2](2) + (2*p[2](0)*x + p[2](2)*(1-x*x))*q[2](0);
			M(2,2) = (p[2](0)*(1-x*x)-2*p[2](2)*x)*q[2](1) - p[2](1)*(1+x*x)*q[2](0);				
			
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
			const Eigen::Vector3d TR = svd.matrixV().col(2);
			
			//find the rectified pose
			Eigen::Matrix3d R = R2.transpose()*RR*R1;
			Eigen::Vector3d T = R2.transpose()*TR;
			//std::cout << evaluate_R(R, Rgt) << " ";
			//std::cout << evaluate_t(T, Tgt) << "\n";
			
			
			//store the solution
			Rs[num_real_sols] = R;
			Ts[num_real_sols] = T;
			++num_real_sols;
			
			//TODO both T and -T are solutions to the system, only one of them should imply positive depth of the point, select the correct solution
		}
	}
	
	//solve the problem for the flipped vanishing point
	const Eigen::MatrixXcd evsf = solver_1vp_3pt(p, qf);
	for(int i=0;i<6;++i)
	{
		if(evsf(i).imag() < 1e-10 && evsf(i).imag() > -1e-10)
		{
			//real solution
			const double x = evsf(i).real();
			
			//build the rotation matrix of the rectified problem
			Eigen::Matrix3d RR;
			RR << (1-x*x)/(1+x*x), 0, -2*x/(1+x*x), 0, 1, 0, 2*x/(1+x*x), 0, (1-x*x)/(1+x*x);
			
			//find the translation of the rectified problem
			Eigen::Matrix3d M;
			M(0,0) = p[0](1)*(1+x*x)*qf[0](2) - (2*p[0](0)*x + p[0](2)*(1-x*x))*qf[0](1);
			M(0,1) = -(p[0](0)*(1-x*x)-2*p[0](2)*x)*qf[0](2) + (2*p[0](0)*x + p[0](2)*(1-x*x))*qf[0](0);
			M(0,2) = (p[0](0)*(1-x*x)-2*p[0](2)*x)*qf[0](1) - p[0](1)*(1+x*x)*qf[0](0);
			
			M(1,0) = p[1](1)*(1+x*x)*qf[1](2) - (2*p[1](0)*x + p[1](2)*(1-x*x))*qf[1](1);
			M(1,1) = -(p[1](0)*(1-x*x)-2*p[1](2)*x)*qf[1](2) + (2*p[1](0)*x + p[1](2)*(1-x*x))*qf[1](0);
			M(1,2) = (p[1](0)*(1-x*x)-2*p[1](2)*x)*qf[1](1) - p[1](1)*(1+x*x)*qf[1](0);
			
			M(2,0) = p[2](1)*(1+x*x)*qf[2](2) - (2*p[2](0)*x + p[2](2)*(1-x*x))*qf[2](1);
			M(2,1) = -(p[2](0)*(1-x*x)-2*p[2](2)*x)*qf[2](2) + (2*p[2](0)*x + p[2](2)*(1-x*x))*qf[2](0);
			M(2,2) = (p[2](0)*(1-x*x)-2*p[2](2)*x)*qf[2](1) - p[2](1)*(1+x*x)*qf[2](0);				
			
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
			const Eigen::Vector3d TR = svd.matrixV().col(2);
			
			//find the rectified pose
			Eigen::Matrix3d R = R2f.transpose()*RR*R1;
			Eigen::Vector3d T = R2f.transpose()*TR;
			//std::cout << evaluate_R(R, Rgt) << " ";
			//std::cout << evaluate_t(T, Tgt) << "\n";
			
			
			//store the solution
			Rs[num_real_sols] = R;
			Ts[num_real_sols] = T;
			++num_real_sols;
			
			//TODO both T and -T are solutions to the system, only one of them should imply positive depth of the point, select the correct solution
		}
	}
	
	return num_real_sols;
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
	
	//GENERATE 1 VANISHING POINT
	Eigen::Vector3d vp1;
	Eigen::Vector3d vq1;
	sample_vp(R, T, vp1, vq1);
	vps[0] = vp1;
	vqs[0] = vq1;
	
	//GENERATE 3 FINITE POINTS (mean [0 0 5], std 1)
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
	
	Eigen::Vector3d p3 = X3/X3(0);
	Eigen::Vector3d q3 = R*X3+T;
	q3 = q3/q3(0);
	pts[2] = p3;
	qts[2] = q3;
}

