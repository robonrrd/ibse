//
//
#include <ibse.h>
#include <ibse_impl.h>
#include <ibse_internal.h>

namespace ibse
{

ErrorCode
Impl::setVisualData(const std::vector<double>& time,
                        const std::vector<Eigen::Vector3d>& position,
                        const std::vector<Eigen::Quaterniond>& rotation)
{
   if ((time.size() != position.size()) || (position.size() != rotation.size()))
   {
      return ErrorCode::ERROR_INCONSISTENT_DATA_SIZES;
   }

   _visTime = time;
   // Normalize time data
   double t0 = _visTime[0];
   for (auto& tt : _visTime)
      tt = tt - t0;

   _visDt = 0;
   const size_t szTime = _visTime.size();
   for (size_t ii=1; ii<szTime; ++ii)
      _visDt += (_visTime[ii] - _visTime[ii-1]);
   _visDt = _visDt / (szTime-1);

   _visPosition = position;

   // Copy the rotations into an ArrayXX for faster processing later
   const size_t szRot = rotation.size();
   _visQuat.resize(szRot, 4);
   for (size_t ii=0; ii<szRot; ++ii)
      _visQuat.row(ii) << rotation[ii].w(), rotation[ii].x(), rotation[ii].y(), rotation[ii].z();

   _state = InternalState::DATA_SET;
   return ErrorCode::SUCCESS;
}


ErrorCode
Impl::setInertialData(const std::vector<double>& accelerometer_time,
                      const std::vector<Eigen::Vector3d>& linear_acceleration,
                      const std::vector<double>& gyroscope_time,
                      const std::vector<Eigen::Vector3d>& angular_velocity)
{
   if ((accelerometer_time.size() != linear_acceleration.size()) ||
       (gyroscope_time.size() != angular_velocity.size()))
   {
      return ErrorCode::ERROR_INCONSISTENT_DATA_SIZES;
   }

   // Copy the linear accelerations into an ArrayXX for faster processing later
   const size_t szAcc = linear_acceleration.size();
   _imuAccleration.resize(szAcc, 3);
   for (size_t ii=0; ii<linear_acceleration.size(); ++ii)
      _imuAccleration.row(ii) = linear_acceleration[ii];

   // Resample gyroscope readings to match the sampling of the accelerometer
   //
   double t0 = std::min(accelerometer_time[0], gyroscope_time[0]);
   _imuTime.clear();
   for (const auto& tt : accelerometer_time)
      _imuTime.push_back(tt - t0);

   const size_t szGyrTime = gyroscope_time.size(); 
   Eigen::VectorXd timeGyr(szGyrTime);
   for (size_t ii=0; ii<szGyrTime; ++ii)
      timeGyr(ii) = gyroscope_time[ii] - t0;

   const size_t szGyr = angular_velocity.size();
   Eigen::VectorXd gyrImu_x(szGyr);
   Eigen::VectorXd gyrImu_y(szGyr);
   Eigen::VectorXd gyrImu_z(szGyr);
   for (size_t ii=0; ii<szGyr; ++ii)
   {
      gyrImu_x(ii) = angular_velocity[ii].x();
      gyrImu_y(ii) = angular_velocity[ii].y();
      gyrImu_z(ii) = angular_velocity[ii].z();
   }

   _imuAngVel.resize(_imuTime.size(), 3);
   _imuAngVel.col(0) = linearInterpolation(timeGyr, gyrImu_x, _imuTime);
   _imuAngVel.col(1) = linearInterpolation(timeGyr, gyrImu_y, _imuTime);
   _imuAngVel.col(2) = linearInterpolation(timeGyr, gyrImu_z, _imuTime);


   _state = InternalState::DATA_SET;
   return ErrorCode::SUCCESS; 
};


ErrorCode
Impl::initialAlignmentEstimation()
{
   if (_state == InternalState::NOT_STARTED)
   {
      return ErrorCode::ERROR_NO_DATA;
   }

   // Kalman & RTS filtering
   Eigen::ArrayXXd visAcc = KalmanRTS(_visPosition, _visDt).transpose();

   // Estimate visual and IMU alignment
   double          td;
   Eigen::Vector3d bg;
   estimateAlignment(_visQuat, _visTime, _imuAngVel, _imuTime, _Rs, td, bg);

   // Align camera and IMU measurements
   Eigen::ArrayXXd out_accVis;
   Eigen::ArrayXXd out_qtVis;
   Eigen::MatrixXd out_accImu;
   alignCameraAndIMU(visAcc, _visQuat, _visTime, _imuAccleration, _imuTime, _Rs, td,
                     out_accVis, out_qtVis, out_accImu, _alignTime);

   // Transform visual accelerations from world frame to local frame
   Eigen::ArrayXXd rotAccVis = rotatePoints(out_qtVis, out_accVis);

   // Find initial estimates for the scale, gravity and bias by solving
   // an unconstrained linear system of equations Ax = b
   initializeEstimates(rotAccVis, out_qtVis, out_accImu, _A, _b, _scale0, _g0, _bias0);

   _state = InternalState::INITIAL_ALIGNMENT;
   return ErrorCode::SUCCESS; 
}


ErrorCode
Impl::estimateScale(double& scale, Eigen::Vector3d& g, Eigen::Vector3d& bias,
                    double fMax)
{
   if (_state == InternalState::NOT_STARTED)
   {
      return ErrorCode::ERROR_NO_DATA;
   }
   else if (_state == InternalState::DATA_SET)
   {
      return ErrorCode::ERROR_NO_INITIAL_ALIGNMENT;
   }

   // Final estimation in the frequency domain
   bool success = optimizeEstimate(_A, _b, _scale0, _g0, _bias0, _alignTime, fMax, 
                                   scale, g, bias);
   
   if (success)
   {
      _state = InternalState::SCALE_ESTIMATED;
      return ErrorCode::SUCCESS; 
   }
   else
   {
      return ErrorCode::ERROR_FINAL_ESTIMATION_FAILED;
   }
}

}; // namespace ibse
