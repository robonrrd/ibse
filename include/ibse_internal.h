#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/Splines>
#include <unsupported/Eigen/FFT>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <nlopt.h>


namespace ibse
{


// Discrete Linear Time-Invariant ODE with possible Gaussian noise
//
// INPUT:    F   : 3x3 Feedback matrix
//           L   : 3x1 Noise effect matrix
//           Qc  : Scalar spectral density
//           dt  : time-step
//
// OUTPUT:   A   : Transition matrix
//           Q   : Discrete process covariance matrix
//
void
ltiDisc(const Eigen::Matrix3d& F, const Eigen::Vector3d& L, double qc, double dt, 
        Eigen::Matrix3d& A, Eigen::Matrix3d& Q);


// Find the relative rotation between the camera and IMU when using the 
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
                double& f);


// One-dimensional linear interpolator. This class is extremely slow and should not be
// used. We keep it to use as a standard implementation, so we can test out own,
// faster, liner interpolation code
class LinearSplineFunction
{
  public:
   LinearSplineFunction(Eigen::VectorXd const &x,   Eigen::VectorXd const &y);

   double operator()(double x) const;

 private:
   // Helpers to scale X values down to [0, 1]
   double scaled_value(double x) const;

   Eigen::RowVectorXd scaled_values(Eigen::VectorXd const &x_vec) const;

   // Minimum and maximum of original data
   double x_min;
   double x_max;
   // Spline of one-dimensional "points."
   Eigen::Spline<double, 1, 1> spline_;
};


// Calculate a variable-sized moving window average of the input data
// INPUT :   in  : The input vector (Nx1 matrix)
//           ww  : Window size
//
// OUTPUT :        Averaged data
//
Eigen::MatrixXd 
movingAverage(const Eigen::MatrixXd& in, const int& ww);


// Fast, single-pass linear interpolator
// INPUT :   X    : Iriginal sampling along the X-axis (e.g. the time of a time-series)
//           Y    : Original data, sampled at values of 'X'
//           X_I  : The interpolation sampling (can include values before and after the
//                  limits of X)
// OUTPUT:          Interpolated values of Y along X_I
//
Eigen::VectorXd 
linearInterpolation(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                    const Eigen::VectorXd& x_i);

Eigen::VectorXd 
linearInterpolation(const Eigen::VectorXd& x, const Eigen::VectorXd& y, 
                    const std::vector<double>& x_i);


// Data structure used by nlopt to pass data to the objective function
struct OptimizationData
{
   // Data passed to objective function
   const Eigen::MatrixXd* A;  // Nx7
   const cv::Mat*         Fi; // precomputed FFT of b
   size_t                 szFreqRange;

   // Status data set within the objective function
   int                    numIters;
}; 


// Objective function, implemented with Eigen's FFT
//
double
freqDomainObjective(unsigned int n, const double* x, double* grad, void* data);


// Utility function to computes the DFT of each row of the input matrix independently, 
// and returns the absolute value of the complex output
//
cv::Mat
abs_dft(const cv::Mat& in);


// Objective function, implemented with OpenCV's DFT. Much faster than Eigen
//
double
opencv_freqDomainObjective(unsigned int n, const double* x, double* grad, void* data);


// Calculate a smooth estimate of the second derivative (accelerations) of 
// the input position data.  We use a Kalman filter, followed by a 
// Rauch–Tung–Striebel filter to perform the smoothing
// 
// INPUT:    pose   : A series of poses, evenly spaced in time
//           dt     : The interval between samples in 'pose'
//
// OUTPUT:          : Smoothed estimates of the acceleratio of the input
//
Eigen::ArrayXXd 
KalmanRTS(const std::vector<Eigen::Vector3d>& pose, double dt);


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
estimateAlignment(const Eigen::ArrayXXd& in_qtVis,
                  const std::vector<double>& in_tVis,
                  const Eigen::ArrayXXd& in_angImu,
                  const std::vector<double>& in_tImu,
                  Eigen::Matrix3d& Rs, double& td, Eigen::Vector3d& bg);


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
alignCameraAndIMU(const Eigen::ArrayXXd& in_accVis,
                  const Eigen::ArrayXXd& in_qtVis, 
                  const std::vector<double>& in_tVis,
                  const Eigen::ArrayXXd& in_accImu,
                  const std::vector<double>& in_tImu,
                  const Eigen::Matrix3d& Rs, const double& td,
                  Eigen::ArrayXXd& out_accVis, Eigen::ArrayXXd& out_qtVis,
                  Eigen::MatrixXd& out_accImu, std::vector<double>& out_t);


// Rotate point p according to unit quaternion q.
//
// INPUT:  q : Unit quaternion(s) as Nx4 Array
//         p : Points as Nx3 arrau
//
// OUTPUT: Rotated point(s) as Nx3 array
//
Eigen::ArrayXXd
rotatePoints(const Eigen::ArrayXXd& quats, const Eigen::ArrayXXd& pts);


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
initializeEstimates(const Eigen::ArrayXXd& accVis,
                    const Eigen::ArrayXXd& qtVis,
                    const Eigen::ArrayXXd& accImu,
                    Eigen::MatrixXd& A, Eigen::MatrixXd& b, double& scale,
                    Eigen::Vector3d& g, Eigen::Vector3d& bias);


// The final scale estimation optimization problem.  We refine the initial
// estimates of the scale, bias, and gravity, by minimizing the difference
// between the visual and inertial accelerations, while maintaining the
// constraint that gravity should be 9.8 m/s^2
//
// INPUT:    A        : Matrix of visual accelerations and rotations (3*Nx7 matrix)
//           b        : Row-matrix of IMU accelerations (3*Nx1 matrix)
//           scale0   : Initial estimate at scale
//           gravity0 : Initial gravity vector estimate (3x1 vector)
//           bias0    : Initial estimate at gyroscope bias(3x1 vector)
//           t        : Vector of times (from 'alignCameraAndIMU')
//           fMax     : The maximum frequency that we care about when
//                      optimizing
//
// OUTPUT:   scale    : Final scale
//           g        : Final gravity (3x1 vector)
//           bias     : Final gyroscope bias (3x1 vector)
//
bool
optimizeEstimate(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b,
                 const double& scale0, const Eigen::Vector3d& g0,
                 const Eigen::Vector3d& bias0, const std::vector<double>& t,
                 const double& fMax,
                 double& scale, Eigen::Vector3d& g, 
                 Eigen::Vector3d& bias);

}; // namespace ibse
