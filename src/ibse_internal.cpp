#include <ibse_internal.h>

namespace ibse
{

// Discrete Linear Time-Invariant ODE with possible Gaussian noise
//
// Input variables:
//   F  - 3x3 Feedback matrix
//   L  - 3x1 Noise effect matrix
//   Qc - Scalar spectral density
//   dt - time-step
//
// Output:
//   A - Transition matrix
//   Q - Discrete process covariance matrix
//
// Description:
//   The original ODE model is of the form
//
//     dx/dt = F x + L w,  w ~ N(0,Qc)
//
// Discretization of the model gives us:
//
//     x[k] = A x[k-1] + q, q ~ N(0,Q)
//
//  Which can be used for integrating the model exactly over time
//  steps, which are multiples of dt.
//
void
ltiDisc(const Eigen::Matrix3d& F, const Eigen::Vector3d& L, double qc, 
        double dt, Eigen::Matrix3d& out_A, Eigen::Matrix3d& out_Q)
{
   using namespace Eigen;

   // Closed form integration of the covariance by matrix fraction decomposition
   MatrixXd Phi(6,6);
   Phi.block<3,3>(0,0) = F;
   Phi.block<3,3>(0,3) = L*qc*L.transpose();
   Phi.block<3,3>(3,0) = Matrix3d::Zero();
   Phi.block<3,3>(3,3) = -F.transpose();

   out_A = (F*dt).exp();

   MatrixXd zeroone(6,3);
   zeroone.block<3,3>(0,0) = Matrix3d::Zero();
   zeroone.block<3,3>(3,0) = Matrix3d::Identity();

   MatrixXd AB(6,3);
   AB = (Phi*dt).exp()*zeroone;
   out_Q = AB.block<3,3>(0,0)*(AB.block<3,3>(3,0).inverse());
}


// Finds the relative rotation between the camera and IMU when using the 
// provided time offset td. Gyroscope bias is estimated in the process.
//
// INPUT:    angVis  : Visual angular velocities [rad/s] (Mx3 matrix)
//           angImu  : Inertial angular velocities [rad/s] (Mx3 matrix)
//           t       : Timestamps in seconds
//           td      : Time offset in seconds
//
// OUTPUT:   Rs      : Rotation between the camera and IMU (3x3 matrix)
//           bias    : Gyroscope bias [rad/s] (1x3 vector)
//           f       : Function value (sum of squared differences)
//
void
solveClosedForm(const Eigen::MatrixXd& angVis,
                const Eigen::MatrixXd& angImu,
                const std::vector<double>& t, double td,
                Eigen::Matrix3d& Rs,
                Eigen::MatrixXd& bias,
                double& f)
{
   using namespace Eigen;

   assert(angVis.rows() == angImu.rows());
   assert(angVis.rows() == (int) t.size());

   const size_t N = angVis.rows();

   // Adjust visual angular velocities based on current offset
   //  * we pretend the angVis data are collected -td before
   //    their timestamps, and then linearlly interpolate quaternions
   //    at T=t1-td and T=t2-td to find the value at T=t1
   //  * we linearly interpret euler angles. This is bad and it
   //    really should be slerping quaternions
   VectorXd t_td(N);
   for (size_t ii=0; ii<t.size(); ++ii)
      t_td(ii) = t[ii] - td;

   VectorXd angX = angVis.col(0);
   VectorXd angY = angVis.col(1);
   VectorXd angZ = angVis.col(2);

   MatrixXd newAng(N, 3);
   MatrixXd meanVis = MatrixXd::Zero(1,3);
   newAng.col(0) = linearInterpolation(t_td, angX, t);
   newAng.col(1) = linearInterpolation(t_td, angY, t);
   newAng.col(2) = linearInterpolation(t_td, angZ, t);
   meanVis(0,0) = newAng.col(0).mean();
   meanVis(0,1) = newAng.col(1).mean();
   meanVis(0,2) = newAng.col(2).mean();

   MatrixXd meanImu = MatrixXd::Zero(1,3);
   for (size_t ii=0; ii<N; ++ii)
      meanImu += angImu.row(ii);
   meanImu = meanImu / N;

   // Center the point sets
   MatrixXd cenAngImu(N, 3);
   for (size_t ii=0; ii<N; ++ii)
      cenAngImu.row(ii) = angImu.row(ii) - meanImu;

   MatrixXd cenAngVis(N, 3);
   for (size_t ii=0; ii<N; ++ii)
      cenAngVis.row(ii) = newAng.row(ii) - meanVis;

   // NOTE: May be slow for large matrices; if so, try BDCSVD
   JacobiSVD<MatrixXd> svd(cenAngImu.transpose()*cenAngVis, 
                           ComputeThinU | ComputeThinV);

   // Ensure a right-handed coordinate system and correct if necessary
   Matrix3d C = Matrix3d::Identity();
   if ((svd.matrixV()*svd.matrixU().transpose()).determinant() < 0)
      C(2,2) = -1;

   // OUTPUT: Rotation between camera and IMU
   Rs =  svd.matrixV()*C*svd.matrixU().transpose();

   // OUTPUT: The gyroscope bias is the translation
   bias = meanVis - meanImu*Rs;

   // Residual
   MatrixXd D = newAng - (angImu*Rs + bias.replicate(N,1));
   
   // OUTPUT
   f = D.array().square().sum();
}


// "Direct Form II Transposed" implementation of the standard difference 
//  equation, assuming 'b' vector is all ones
std::vector<double>
df2t(const Eigen::MatrixXd& x, size_t ww)
{
   const int x_sz = x.rows();
   std::vector<double> y(x_sz,0);
   for (int ii=0; ii<x_sz; ++ii)
   {
      y[ii] = 0;
      for (size_t jj=0; jj<ww; ++jj)
      {
         int idx = ii-jj;
         if (idx >= 0)
            y[ii] += x(idx,0);
      }
      y[ii] /= (float)ww;
   }
   return y;
}


// One-dimensional linear interpolator
LinearSplineFunction::LinearSplineFunction(Eigen::VectorXd const &x_vec,
                                           Eigen::VectorXd const &y_vec)
   : x_min(x_vec.minCoeff()),
     x_max(x_vec.maxCoeff()),

     // Spline fitting here. X values are scaled down to [0, 1] for this.
     spline_(Eigen::SplineFitting<Eigen::Spline<double, 1, 1>>::Interpolate(
                y_vec.transpose(),
                1, // degree
                scaled_values(x_vec)))
{ }

double
LinearSplineFunction::operator()(double x) const
{
   // x values need to be scaled down in extraction as well.
   return spline_(scaled_value(x))(0);
}

// Helpers to scale X values down to [0, 1]
double
LinearSplineFunction::scaled_value(double x) const
{
   return (x - x_min) / (x_max - x_min);
}

Eigen::RowVectorXd
LinearSplineFunction::scaled_values(Eigen::VectorXd const &x_vec) const
{
   return x_vec.unaryExpr([this](double x)
                          { return scaled_value(x); }).transpose();
}


Eigen::VectorXd 
linearInterpolation(const Eigen::VectorXd& x, const Eigen::VectorXd& y, 
                    const Eigen::VectorXd& x_i)
{
   const size_t sz_x = x.size();
   const size_t sz_xi = x_i.size();
   Eigen::VectorXd y_i(sz_xi);

   size_t i0 = 0; 
   size_t i1 = 1;
   for (size_t ii=0; ii<sz_xi; ++ii)
   {
       assert(x(i1) > x(i0)); // enforce strict monotonicity

      // Find the first value of X that is larger than our interpolation query
      //  value, x_i(ii)
      while ((x_i(ii) > x(i1)) && (i1 < sz_x-1))
         i1++;
      i0 = i1;

      // Find the first value of X that is smaller  than our interpolation query
      //  value
      while ((x_i(ii) < x(i0)) && (i0 > 0))
         i0--;
      // ..ensure that they are not the same during end extrapolation
      if (i0 == i1)
         i0--;

      y_i(ii) = y(i0) + (x_i(ii) - x(i0))*( (y(i1)-y(i0))/(x(i1)-x(i0)) );
   }

   return y_i;
}

Eigen::VectorXd 
linearInterpolation(const Eigen::VectorXd& x, const Eigen::VectorXd& y, 
                    const std::vector<double>& x_i)
{
   const size_t sz_x = x.size();
   const size_t sz_xi = x_i.size();
   Eigen::VectorXd y_i(sz_xi);

   size_t i0 = 0; 
   size_t i1 = 1;
   for (size_t ii=0; ii<sz_xi; ++ii)
   {
       assert(x(i1) > x(i0)); // enforce strict monotonicity

      // Find the first value of X that is larger than our interpolation query
      //  value, x_i(ii)
      while ((x_i[ii] > x(i1)) && (i1 < sz_x-1))
         i1++;
      i0 = i1;

      // Find the first value of X that is smaller  than our interpolation query
      //  value
      while ((x_i[ii] < x(i0)) && (i0 > 0))
         i0--;
      // ..ensure that they are not the same during end extrapolation
      if (i0 == i1)
         i0--;

      y_i(ii) = y(i0) + (x_i[ii] - x(i0))*( (y(i1)-y(i0))/(x(i1)-x(i0)) );
   }

   return y_i;
}



// Adapting moving window average
//
Eigen::MatrixXd
movingAverage(const Eigen::MatrixXd& in, const int& ww)
{
   Eigen::MatrixXd out(in.rows(), in.cols());
   int row_ct = 0;

   std::vector<double> y_filt = df2t(in, ww);
   
   std::vector<double> begin(ww-2);
   begin[0] = in(0,0);
   for (int ii=1; ii<ww-2; ++ii)
      begin[ii] = in(ii,0) + begin[ii-1];
   
   for (int ii=0; ii<ww-2; ii+=2, row_ct++)
      out(row_ct,0) = begin[ii] / (ii+1);
   
   for (size_t ii=ww-1; ii<y_filt.size(); ++ii, ++row_ct)
      out(row_ct,0) = y_filt[ii];
   
   const size_t sz_in = in.size()-1;
   std::vector<double> end;
   end.push_back( in(in.rows()-1,0) );
   for (size_t ii=sz_in-1; ii>=(sz_in-ww+3); --ii)
      end.push_back(in(ii,0) + end.back());
   
   for (int ii=end.size()-1, jj=0; ii>=0 ;ii-=2, jj+=2, row_ct++)
      out(row_ct,0) = end[ii] / (ww-2-jj);
   
   return out;
}


///


double
freqDomainObjective(unsigned int n, const double* x, double* grad, void* data)
{
   assert(n == 7);
    
   using namespace Eigen;

   OptimizationData* pData = static_cast<OptimizationData*>(data);
   pData->numIters++;

   
   Map<const RowVectorXd> vx(x, pData->A->cols()); // 7x1 vector
   MatrixXd eAv = *(pData->A)*vx.transpose(); // Nx1
   const size_t N = eAv.rows()/3; 
   std::vector<double> AvX(N);
   std::vector<double> AvY(N);
   std::vector<double> AvZ(N);
   for (size_t ii=0; ii<N; ++ii)
   {
      AvX[ii] = eAv(3*ii,0);
      AvY[ii] = eAv(3*ii+1,0);
      AvZ[ii] = eAv(3*ii+2,0);
   }

   static FFT<double> fft;
   std::vector<std::complex<double>> Fx;
   std::vector<std::complex<double>> Fy;
   std::vector<std::complex<double>> Fz;
   fft.fwd(Fx, AvX);
   fft.fwd(Fy, AvY);
   fft.fwd(Fz, AvZ);

   ArrayXXd Fv(pData->szFreqRange, 3);
   for (size_t ii=0; ii<pData->szFreqRange; ++ii)
      Fv.row(ii) << std::abs(Fx[ii]), std::abs(Fy[ii]), std::abs(Fz[ii]);

   Fx.clear();
   Fy.clear();
   Fz.clear();

   ArrayXXd Fi(pData->szFreqRange, 3);
   for (size_t ii=0; ii<pData->szFreqRange; ++ii)
       Fi.row(ii) << pData->Fi->at<double>(0,ii), pData->Fi->at<double>(1,ii),
           pData->Fi->at<double>(2,ii);

   ArrayXXd diff = Fv-Fi;
   return (diff*diff).sum();
}


// Computes the DFT of each row of the input matrix independently, and returns
// the absolute value of the complex output
cv::Mat
abs_dft(const cv::Mat& in)
{
    using namespace cv;

    Mat cmplx_out;
    dft(in, cmplx_out, DFT_ROWS | DFT_COMPLEX_OUTPUT);

    Mat re_im[2];
    split(cmplx_out, re_im);
    
    re_im[0] = re_im[0].mul(re_im[0]);
    re_im[1] = re_im[1].mul(re_im[1]);
    
    Mat out(in.rows, in.cols, CV_64F);
    sqrt(re_im[0]+re_im[1], out);
    
    return out;
}


double
opencv_freqDomainObjective(unsigned int n, const double* x, double* grad, void* data)
{
   using namespace Eigen;
   
   assert(n == 7);

   OptimizationData* pData = static_cast<OptimizationData*>(data);

   if (grad)
   {
       const double eps = 1e-6;
       double* up_x = new double[n];
       double* down_x = new double[n];

       memcpy(up_x,   x, n*sizeof(double));
       memcpy(down_x, x, n*sizeof(double));

       for (size_t ii=0; ii<n; ++ii)
       {
           up_x[ii]   = x[ii] + eps;
           down_x[ii] = x[ii] - eps;

           double up   = opencv_freqDomainObjective(n, up_x, 0, data);
           double down = opencv_freqDomainObjective(n, down_x, 0, data);
           pData->numIters -= 2; // correction of iteration counter

           grad[ii] = (up-down)/(2*eps);

           up_x[ii]   = x[ii];
           down_x[ii] = x[ii];
       }

       delete[] up_x;
       delete[] down_x;
   }
    
   pData->numIters++;
   
   Map<const RowVectorXd> vx(x, pData->A->cols()); // 7x1 vector
   MatrixXd eAv = *(pData->A)*vx.transpose(); // Nx1
   const size_t N = eAv.rows()/3; 
   const size_t N_opt = cv::getOptimalDFTSize(N);

   cv::Mat Av = cv::Mat::zeros(3, N_opt, CV_64F);
   for (size_t ii=0; ii<N; ++ii)
      for (size_t jj=0; jj<3; ++jj)
         Av.at<double>(jj, ii) = eAv(3*ii+jj, 0);

   cv::Mat Fv = abs_dft(Av)(cv::Range(0, 3), cv::Range(0, pData->szFreqRange));

   cv::Mat diff = Fv - *(pData->Fi);
   return cv::sum( diff.mul(diff) )[0];
}




Eigen::ArrayXXd
KalmanRTS(const std::vector<Eigen::Vector3d>& pose, double dt)
{
   using namespace Eigen;
   
   // Assume that we have roughly uniform sampling to optimize computations,
   //  otherwise we would need to recompute A and Q in the loop which is
   //  slower. 
   const size_t numPts = pose.size();

   ArrayXXd acc_kfs(3, numPts);

   Matrix3d F;
   F << 0, 1, 0,
        0, 0, 1,
        0, 0, 0;
   double R = 0.01;
   Vector3d L(0,0,1);
   Vector3d H(1,0,0);
   Vector3d m0(0,0,0);
   Matrix3d P0 = Matrix3d::Identity() * 1e4;

   // Compute maximum likelihood estimate of qc on a grid
   std::vector<double> qc_list = {45, 55, 65, 75, 85, 95, 105, 115, 125};
   std::vector<double> lh_list = {0, 0, 0, 0, 0, 0, 0, 0, 0};

   ArrayXXd pos(3, numPts);
   for (size_t pp=0; pp<numPts; ++pp)
   {
      pos(0, pp) = pose[pp].x();
      pos(1, pp) = pose[pp].y();
      pos(2, pp) = pose[pp].z();
   }

   const size_t numQc = qc_list.size();
   for (size_t jj=0; jj<numQc; ++jj)
   {
      double qc = qc_list[jj];

      Matrix3d A;
      Matrix3d Q;

      ltiDisc(F, L, qc, dt, A, Q);

      double lh = 0;
      for (size_t ii=0; ii<3; ++ii)
      {
         // Kalman filter
         Vector3d m = m0;
         Matrix3d P = P0;
         MatrixXd kf_m = MatrixXd::Zero(3, numPts);
         std::vector<Matrix3d> kf_P(numPts);

         for (size_t kk=0; kk<numPts; ++kk)
         {
             m = A*m;
             P = A*P*A.transpose() + Q;

             double nu = pos(ii, kk) - H.dot(m);
             double S = H.transpose()*P*H + R;
             Vector3d K = (P*H) / S;

             m = m + K*nu;
             P = P - K*S*K.transpose();

             lh = lh + 0.5*log(2*M_PI) + 0.5*log(S) + 0.5*nu*nu/S;

             kf_m.block<3,1>(0,kk) = m;
             kf_P[kk] = P;
         }

      }
      lh_list[jj] = lh;
   }

   size_t idx = std::min_element(lh_list.begin(), lh_list.end()) - lh_list.begin();
   double qc = qc_list[idx];

   // Kalman filter and smoother
   Matrix3d A;
   Matrix3d Q;
   ltiDisc(F, L, qc, dt, A, Q);

   for (size_t ii=0; ii<3; ++ii)
   {
      // Kalman filter
      Vector3d m = m0;
      Matrix3d P = P0;
      MatrixXd kf_m = MatrixXd::Zero(3, numPts);
      std::vector<Matrix3d> kf_P(numPts);
      
      for (size_t kk=0; kk<numPts; ++kk)
      {
         m = A*m;
         P = A*P*A.transpose() + Q;

         double S = H.transpose()*P*H + R;
         Vector3d K = (P*H) / S;

         m = m + K*(pos(ii, kk) - H.transpose()*m);
         P = P - K*S*K.transpose();
         
         kf_m.block<3,1>(0,kk) = m;
         kf_P[kk] = P;
      }

      // Rauch–Tung–Striebel smoother
      Vector3d ms = m;
      Matrix3d Ps = P;
      MatrixXd rts_m = MatrixXd::Zero(3, numPts);
      rts_m.block<3,1>(0, numPts-1) = ms;
      std::vector<Matrix3d> rts_P(numPts);
      rts_P.back() = Ps;
      
      for (int kk=numPts-2; kk>=0; --kk)
      {
         MatrixXd mp(3,1);
         mp = A*kf_m.block<3,1>(0,kk);

         Matrix3d Pp = A*kf_P[kk]*A.transpose() + Q;
         Matrix3d Ck = kf_P[kk]*A.transpose()*Pp.inverse();
         ms = kf_m.block<3,1>(0,kk) + Ck*(ms - mp);
         Ps = kf_P[kk] + Ck*(Ps - Pp)*Ck.transpose();
         rts_m.block<3,1>(0, kk) = ms;
         rts_P[kk] = Ps;
      }
      
      acc_kfs.row(ii) = rts_m.row(2).array();
   }
   
   return acc_kfs;
}


// Estimate temporal and spatial alignment between the camera and IMU.
// Gyroscope bias is also estimated in the process.
//
// INPUT:    qtVis   : Visual orientations (Nx4 matrix)
//           tVis    : Visual timestamps in seconds (Nx1 vector)
//           angImu  : Inertial angular velocities [rad/s] (Mx3 matrix)
//           tImu    : Inertial timestamps in seconds (Mx1 vector)
//
// OUTPUT:   Rs      : Rotation between the camera and IMU (3x3 matrix)
//           td      : Time offset between the camera and IMU (scalar)
//           bg      : Gyroscope bias [rad/s] (1x3 vector)
//
// RETURN:   Number of golden-section search iterations
//
int
estimateAlignment(const Eigen::ArrayXXd& in_qtVis, const std::vector<double>& in_tVis,
                  const Eigen::ArrayXXd& in_angImu, const std::vector<double>& in_tImu,
                  Eigen::Matrix3d& Rs, double& td, Eigen::Vector3d& bg)
{
   assert(in_qtVis.rows() == (int) in_tVis.size());
   assert(in_angImu.rows() == (int) in_tImu.size());

   using namespace Eigen;

   // Only use time spans where both sensors have values
   const double timeStop = std::min(in_tVis.back(), in_tImu.back());

   // Truncate input that extends beyond timeStop
   size_t vis_sz = in_tVis.size();
   for (size_t ii=0; ii<in_tVis.size(); ++ii)
      if (timeStop < in_tVis[ii])
      {
         vis_sz = ii;
         break;
      }
   
   size_t imu_sz = in_tImu.size();
   for (size_t ii=0; ii<in_tImu.size(); ++ii)
      if (timeStop < in_tImu[ii])
      {
         imu_sz = ii;
         break;
      }

   std::vector<double> tVis(&in_tVis[0], &in_tVis[vis_sz]);
   std::vector<double> tImu(&in_tImu[0], &in_tImu[imu_sz]);

   ArrayXXd qtVis = in_qtVis.block(0,0,  vis_sz, 4);
   ArrayXXd angImu = in_angImu.block(0,0,  imu_sz, 3);

   // Upsample visual data to match the sampling of the IMU
   double dt = 0.0;
   for (size_t ii=1; ii<imu_sz; ++ii)
      dt += tImu[ii] - tImu[ii-1];
   dt /= (imu_sz-1);

   VectorXd t_vis(vis_sz); // temporary
   for (size_t ii=0; ii<vis_sz; ++ii)
      t_vis(ii) = tVis[ii];


   // NB: We really should be using slerps here, not linear interpolation
   MatrixXd newQtVis(imu_sz, 4);
   newQtVis.col(0) = linearInterpolation(t_vis, qtVis.col(0), tImu);
   newQtVis.col(1) = linearInterpolation(t_vis, qtVis.col(1), tImu);
   newQtVis.col(2) = linearInterpolation(t_vis, qtVis.col(2), tImu);
   newQtVis.col(3) = linearInterpolation(t_vis, qtVis.col(3), tImu);

   // Compute visual angular velocities
   MatrixXd qtDiffs(imu_sz, 4);
   for (size_t ii=1; ii<imu_sz; ++ii)
      qtDiffs.row(ii-1) = newQtVis.row(ii) - newQtVis.row(ii-1);
   qtDiffs.row(imu_sz-1) = qtDiffs.row(imu_sz-2);

   MatrixXd qtAngVis(imu_sz,3);
   const double mm = -2.0/dt;
   for (size_t ii=0; ii<imu_sz; ++ii)
   {
      Quaterniond a(qtDiffs(ii,0), qtDiffs(ii,1), qtDiffs(ii,2), qtDiffs(ii,3));
      Quaterniond b(newQtVis(ii,0), newQtVis(ii,1), newQtVis(ii,2), newQtVis(ii,3));
      Quaterniond c = a*b.inverse();
      qtAngVis.row(ii) << mm*c.x(), mm*c.y(), mm*c.z();
   }

   qtAngVis.col(0) = movingAverage(qtAngVis.col(0), 15);
   qtAngVis.col(1) = movingAverage(qtAngVis.col(1), 15);
   qtAngVis.col(2) = movingAverage(qtAngVis.col(2), 15);

   angImu.col(0) = movingAverage(angImu.col(0), 15);
   angImu.col(1) = movingAverage(angImu.col(1), 15);
   angImu.col(2) = movingAverage(angImu.col(2), 15);

   const double gRatio = (1.0f + sqrt(5.0)) / 2.0;
   const double tolerance = 1e-4;

   double maxOffset = 0.5;
   double a = -maxOffset;
   double b = maxOffset;

   double c = b - (b - a) / gRatio;
   double d = a + (b - a) / gRatio;
   int iter = 0;

   while (fabs(c - d) > tolerance)
   {
      // Evaluate function at f(c) and f(d)
      Eigen::Matrix3d Rsc, Rsd;
      Eigen::MatrixXd biasc, biasd;
      double fc, fd;

      solveClosedForm(qtAngVis, angImu, tImu, c, Rsc, biasc, fc);
      solveClosedForm(qtAngVis, angImu, tImu, d, Rsd, biasd, fd);

      if (fc < fd)
      {
         b = d;
         Rs = Rsc;
         bg = biasc.transpose();
      }
      else
      {
         a = c;
         Rs = Rsd;
         bg = biasd.transpose();
      }
    
      c = b - (b - a) / gRatio;
      d = a + (b - a) / gRatio;

      ++iter;
   }

   td = (b + a) / 2; // output

   return iter;
}


// Align inertial and visual-data both temporarily and spatially.
//
// INPUT:    accVis  : Visual acceleration [unknown scale] (Nx3 matrix)
//           qtVis   : Visual orientations (Nx4 matrix)
//           tVis    : Visual timestamps in seconds (Nx1 vector)
//           accImu  : Inertial accelerations [m/s^2] (Mx3 matrix)
//           tImu    : Inertial timestamps in seconds (Mx1 vector)
//           Rs      : Rotation between the camera and IMU (3x3 matrix)
//           td      : Time offset between the camera and IMU (scalar)
//
// OUTPUT:   accVis  : Aligned visual acceleration (Kx3 matrix)
//           qtVis   : Aligned visual orientations (Kx4 matrix)
//           accImu  : Aligned inertial accelerations [m/s^2] (Kx3 matrix)
//           t       : Timestamps in seconds (Kx1 vector)
//
void
alignCameraAndIMU(const Eigen::ArrayXXd& in_accVis, const Eigen::ArrayXXd& in_qtVis, 
                  const std::vector<double>& in_tVis,
                  const Eigen::ArrayXXd& in_accImu, const std::vector<double>& in_tImu,
                  const Eigen::Matrix3d& Rs, const double& td,
                  Eigen::ArrayXXd& out_accVis, Eigen::ArrayXXd& out_qtVis,
                  Eigen::MatrixXd& out_accImu, std::vector<double>& out_t)
{
   using namespace Eigen;

   // Only use time spans where both sensors have values
   const double timeStop = std::min(in_tVis.back(), in_tImu.back());

   // Truncate input that extends beyond timeStop
   size_t vis_sz = in_tVis.size();
   for (size_t ii=0; ii<in_tVis.size(); ++ii)
      if (timeStop < in_tVis[ii])
      {
         vis_sz = ii;
         break;
      }
   
   size_t imu_sz = in_tImu.size();
   for (size_t ii=0; ii<in_tImu.size(); ++ii)
      if (timeStop < in_tImu[ii])
      {
         imu_sz = ii;
         break;
      }

   std::vector<double> tVis(&in_tVis[0], &in_tVis[vis_sz]);
   std::vector<double> tImu(&in_tImu[0], &in_tImu[imu_sz]);

   
   ArrayXXd accVis = in_accVis.block(0,0,  vis_sz, 3);
   ArrayXXd qtVis = in_qtVis.block(0,0,  vis_sz, 4);
   ArrayXXd accImu = in_accImu.block(0,0,  imu_sz, 3);

   // Upsample visual data to match the sampling of the IMU
   VectorXd t_vis(vis_sz); // temporary
   for (size_t ii=0; ii<vis_sz; ++ii)
      t_vis(ii) = tVis[ii] - td;

   const size_t K = tImu.size();
   out_accVis = Eigen::ArrayXXd::Zero(K,3);
   out_accVis.col(0) = linearInterpolation(t_vis, accVis.col(0), tImu);
   out_accVis.col(1) = linearInterpolation(t_vis, accVis.col(1), tImu);
   out_accVis.col(2) = linearInterpolation(t_vis, accVis.col(2), tImu);

   out_qtVis = Eigen::ArrayXXd::Zero(K,4);
   out_qtVis.col(0) = linearInterpolation(t_vis, qtVis.col(0), tImu);
   out_qtVis.col(1) = linearInterpolation(t_vis, qtVis.col(1), tImu);
   out_qtVis.col(2) = linearInterpolation(t_vis, qtVis.col(2), tImu);
   out_qtVis.col(3) = linearInterpolation(t_vis, qtVis.col(3), tImu);

   out_accImu = accImu.matrix()*Rs;

   out_t = tImu;
}


// Rotate point p according to unit quaternion q.
//
// INPUT:  q : Unit quaternion(s) as Nx4 Array
//         p : Points as Nx3 arrau
//
// OUTPUT: Rotated point(s) as Nx3 array
//
Eigen::ArrayXXd
rotatePoints(const Eigen::ArrayXXd& quats, const Eigen::ArrayXXd& pts)
{
    using namespace Eigen;

    assert(quats.rows() == pts.rows());
    const size_t N = pts.rows();

    ArrayXXd out(N, 3);
    for (size_t ii=0; ii<N; ++ii)
    {
        Vector3d pt = Quaterniond(quats(ii,0), quats(ii,1), quats(ii,2), quats(ii,3)) *
            Vector3d(pts(ii,0), pts(ii,1), pts(ii,2));
        out.row(ii) << pt.x(), pt.y(), pt.z();
    }

    return out;
}


// Initial estimates for scale, gravity and IMU bias
//
// INPUT:    accVis  : Visual acceleration [unknown scale] (Nx3 matrix)
//           qtVis   : Visual orientations (Nx4 matrix)
//           accImu  : Inertial accelerations [m/s^2] (Mx3 matrix)
//
// OUTPUT:   A       : Matrix of visual accelerations and rotations (3*Nx7 matrix)
//           b       : Row-matrix of IMU accelerations (3*Nx1 matrix)
//           scale   : Initial estimate at scale
//           gravity : The initial gravity vector estimate (3x1 vector)
//           bias    : Initial estimate at gyroscope bias(3x1 vector)
//
void
initializeEstimates(const Eigen::ArrayXXd& accVis, const Eigen::ArrayXXd& qtVis,
                    const Eigen::ArrayXXd& accImu,
                    Eigen::MatrixXd& A, Eigen::MatrixXd& b, double& scale,
                    Eigen::Vector3d& g, Eigen::Vector3d& bias)
{
   using namespace Eigen;

   const size_t N = accVis.rows();

   A.resize(3*N, 7);
   b.resize(3*N, 1);

   for (size_t nn=0; nn<N; ++nn)
   {
      const size_t rr = 3*nn;

      A.block<3,1>(rr,0) = accVis.row(nn).transpose();
      Quaterniond qq(qtVis(nn,0), qtVis(nn,1), qtVis(nn,2), qtVis(nn,3));
      A.block<3,3>(rr,1) = qq.normalized().toRotationMatrix();
      A.block<3,3>(rr,4) = Matrix3d::Identity();

      b.block<3,1>(rr,0) = accImu.row(nn).transpose();
   }

   // Using Householder rank-revealing QR decomposition of a matrix with 
   // column-pivoting.  This technique provides a food compromise between
   // speed and accuracy:
   //   https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
   VectorXd x = A.colPivHouseholderQr().solve(b);

   scale = x(0);
   g << x(1), x(2), x(3);
   bias << x(4), x(5), x(6);
}

// Inequality constraints are of the form: c(x) <= 0

double
gravityConstraint_hi(unsigned n, const double *x, double *grad, void *data)
{
   assert(n == 7);

   const double norm = sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3]);
   if (grad)
   {
      const double min_sqrt = 1e-10;
      if (fabs(norm) < min_sqrt)
      {
         for (size_t ii=0; ii<n; ++ii)
            grad[ii] = 0.0;
      }
      else
      {
         grad[0] = 0; // no influence on constraint
         grad[1] = -grad[1] / norm;
         grad[2] = -grad[2] / norm;
         grad[3] = -grad[3] / norm;
         grad[4] = 0;
         grad[5] = 0;
         grad[6] = 0;
      }
   }

   // G = X[1-3] should equal 9.8
   return 9.8 - norm;
}


double
gravityConstraint_lo(unsigned n, const double *x, double *grad, void *data)
{
   assert(n == 7);

   const double norm = sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3]);
   if (grad)
   {
      const double min_sqrt = 1e-10;
      if (fabs(norm) < min_sqrt)
      {
         for (size_t ii=0; ii<n; ++ii)
            grad[ii] = 0.0;
      }
      else
      {
         grad[0] = 0; // no influence on constraint
         grad[1] = grad[1] / norm;
         grad[2] = grad[2] / norm;
         grad[3] = grad[3] / norm;
         grad[4] = 0;
         grad[5] = 0;
         grad[6] = 0;
      }
   }

   // G = X[1-3] should equal 9.8
   return norm - 9.8;
}

double
gravityConstraint_eq(unsigned n, const double *x, double *grad, void *data)
{
   assert(n == 7);

   const double norm = sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3]);
   if (grad)
   {
      const double min_sqrt = 1e-10;
      if (fabs(norm) < min_sqrt)
      {
         for (size_t ii=0; ii<n; ++ii)
            grad[ii] = 0.0;
      }
      else
      {
         grad[0] = 0; // no influence on constraint
         grad[1] = grad[1] / norm;
         grad[2] = grad[2] / norm;
         grad[3] = grad[3] / norm;
         grad[4] = 0;
         grad[5] = 0;
         grad[6] = 0;
      }
   }

   // G = X[1-3] should equal 9.8
   return norm - 9.8;
}


// Final scale estimation, done in the frequency domain
//
bool
optimizeEstimate(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, const double& scale0,
                 const Eigen::Vector3d& g0, const Eigen::Vector3d& bias0,
                 const std::vector<double>& t, const double& fMax,
                 double& out_scale, Eigen::Vector3d& out_g, Eigen::Vector3d& out_bias)
{
   const size_t N = t.size();
   
   // TODO: This is used elsewhere; compute once and change function signatures
   double dt = 0;
   for (size_t ii=1; ii<N; ++ii)
      dt += (t[ii] - t[ii-1]);
   double fs = (1.0*(N-1))/dt;

   OptimizationData data;
   data.A = &A;

   data.szFreqRange = 0;
   for (size_t ii=0; ii<(N/2); ++ii)
   {
      double f = fs*((1.0f*ii)/(1.0f*N));
      if (f <= fMax)
         data.szFreqRange++;
      else
         break;
   }

   // Precompute the FFT of 'b'
   const size_t Nb = b.rows() / 3;
   const size_t Nb_opt = cv::getOptimalDFTSize(Nb);

   cv::Mat Ai = cv::Mat::zeros(3, Nb_opt, CV_64F); // Pad Ai with zeros
   for (size_t ii=0; ii<Nb; ++ii)
      for (size_t jj=0; jj<3; ++jj)
         Ai.at<double>(jj, ii) = b(ii*3+jj,0);
   
   cv::Mat Fi = abs_dft(Ai);

   cv::Mat Fit = Fi(cv::Range(0, 3), cv::Range(0, data.szFreqRange));
   data.Fi = &Fit;

   data.numIters = 0;
   
   // Optimize
   //
   const int dim = A.cols();
   assert(dim == 7);
   
   nlopt_opt opt;
   // Algorithm and dimensionality
   opt = nlopt_create(NLOPT_LN_COBYLA, dim);
   //opt = nlopt_create(NLOPT_LD_MMA, dim);
   //opt = nlopt_create(NLOPT_LD_SLSQP, dim);

   // stop when an optimization step (or an estimate of the optimum) changes every
   // parameter by less than tol multiplied by the absolute value of the parameter
   nlopt_set_xtol_rel(opt, 1e-4);
   // stop when an optimization step (or an estimate of the optimum) changes every
   // parameter by less than tol
   nlopt_set_xtol_abs1(opt, 1e-4);

   // stop when an optimization step (or an estimate of the optimum) changes the objective
   // function value by less than tol multiplied by the absolute value of the function
   // value
   nlopt_set_ftol_rel(opt, 1e-6);
   // stop when an optimization step (or an estimate of the optimum) changes the function
   // value by less than tol.
   nlopt_set_ftol_abs(opt, 1e-6);
   
   double* lowerBounds = new double[dim];
   double* upperBounds = new double[dim];
  
   // scale bounds
   const double s_max = std::max(fabs(scale0), 50.0);
   lowerBounds[0] = -s_max; upperBounds[0] = s_max;

   // gravity bounds
   const double g_max = std::max(std::max(std::max(fabs(g0.x()),fabs(g0.y())), fabs(g0.z())), 10.0);
   for (size_t ii=0; ii<3; ++ii)
   {
       lowerBounds[ii+1] = -g_max; upperBounds[ii+1] = g_max;
   }

   // bias bounds
   const double b_max = std::max(std::max(std::max(fabs(bias0.x()),fabs(bias0.y())),
                                          fabs(bias0.z())), 10.0);
   for (size_t ii=0; ii<3; ++ii)
   {
       lowerBounds[ii+4] = -b_max; upperBounds[ii+4] = b_max;
   }
   nlopt_set_lower_bounds(opt, lowerBounds);
   nlopt_set_upper_bounds(opt, upperBounds);

   // Eigen FFT objective function:
   // nlopt_set_min_objective(opt, freqDomainObjective, static_cast<void*>(&data));

   // OpenCV FFT objective function:
   nlopt_set_min_objective(opt, opencv_freqDomainObjective, static_cast<void*>(&data));

   // Add the gravity (norm(g) = 9.8) constraint
   //
   // COBYLA: inequality or equality 
   // MMA:    inequality only
   // SLSQP:  equality

   // Allowed constraint violation
   const double constraint_eps = 1e-4;

   if (nlopt_add_inequality_constraint(opt, gravityConstraint_hi, NULL, constraint_eps) < 0)
      std::cout << "ERROR adding hi inequality constraint!" << std::endl;
   
   if (nlopt_add_inequality_constraint(opt, gravityConstraint_lo, NULL, constraint_eps) < 0)
      std::cout << "ERROR adding lo inequality constraint!" << std::endl;

   //if (nlopt_add_equality_constraint(opt, gravityConstraint_eq, NULL, constraint_eps) < 0)
   //   std::cout << "ERROR adding equality constraint!" << std::endl;

   // Starting point
   double* x = new double[dim];
   x[0] = scale0;
   x[1] = g0.x(); x[2] = g0.y(); x[3] = g0.z();
   x[4] = bias0.x(); x[5] = bias0.y(); x[6] = bias0.z();

   double minf; // the objective function value upon return
   const int err_code = nlopt_optimize(opt, x, &minf);
   if (err_code < 0) 
   {
      std::cout << "nlopt failed!" << std::endl;
      switch (err_code)
      {
         default:
         case -1:
            std::cout << " Generic failure" << std::endl;
            break;
         case -2:
            std::cout << " Invalid arguments" << std::endl;
            break;
         case -3:
            std::cout << " Out of memory" << std::endl;
            break;
         case -4:
            std::cout << " Halted because roundoff errors limited progress" << std::endl;
            std::cout << " obj="  << minf << std::endl;
            std::cout << "  hi violation: " << gravityConstraint_hi(dim,x,0,0) << std::endl;
            std::cout << "  lo violation: " << gravityConstraint_lo(dim,x,0,0) << std::endl;
            break;
         case -5:
            std::cout << " Forced stop" << std::endl;
            break;
      }
   }
   else 
   {
      switch (err_code)
      {
         default:
         case 1:
            std::cout << " Generic success" << std::endl;
            break;
         case 2:
            std::cout << " stopval reached" << std::endl;
            break;
         case 3:
            // 'ftol' is the absolute tolerance on the change in the objective function
            // value. We stop when an optimization step (or an estimate of the optimum)
            // changes the function value by less than 'ftol'.
            std::cout << " ftol reached" << std::endl;
            break;
         case 4:
            // 'xtol' is the relative tolerance on optimization parameters. We stop when
            // an optimization step (or an estimate of the optimum) changes every
            // parameter by less than tol multiplied by the absolute value of the
            // parameter.
            std::cout << " xtol reached" << std::endl;
            break;
         case 5:
            std::cout << " maxeval reached" << std::endl;
            break;
         case 6:
            std::cout << " maxtime reached" << std::endl;
            break;
      }
      
      std::cout << " Found minimum: obj="  << minf << std::endl;
      
      std::cout << "  hi violation: " << gravityConstraint_hi(dim,x,0,0) << std::endl;
      std::cout << "  lo violation: " << gravityConstraint_lo(dim,x,0,0) << std::endl;
   }

   out_scale = x[0];
   out_g = Eigen::Vector3d(x[1], x[2], x[3]);
   out_bias = Eigen::Vector3d(x[4], x[5], x[6]);

   std::cout << " " << data.numIters << " iterations" << std::endl;

   nlopt_destroy(opt);
   delete[] lowerBounds;
   delete[] upperBounds;
   delete[] x;

   return (err_code >= 0);
}

}; // namespace ibse
