//
//

#include <ibse.h>
#include <ibse_impl.h>

namespace ibse
{

IBSE::IBSE()
    : _impl(std::unique_ptr<Impl>(new Impl))
{}


IBSE::~IBSE()
{}


ErrorCode
IBSE::setVisualData(const std::vector<double>& time,
              const std::vector<Eigen::Vector3d>& position,
              const std::vector<Eigen::Quaterniond>& rotation)
{
   return _impl->setVisualData(time, position, rotation);
}


ErrorCode
IBSE::setInertialData(const std::vector<double>& accelerometer_time,
                      const std::vector<Eigen::Vector3d>& linear_acceleration,
                      const std::vector<double>& gyroscope_time,
                      const std::vector<Eigen::Vector3d>& angular_velocity)
{
   return _impl->setInertialData(accelerometer_time, linear_acceleration,
                                 gyroscope_time, angular_velocity);
}


ErrorCode
IBSE::initialAlignmentEstimation()
{
   return _impl->initialAlignmentEstimation();
}


ErrorCode
IBSE::estimateScale(double& scale, Eigen::Vector3d& g, Eigen::Vector3d& bias,
                    double fMax)
{
   return _impl->estimateScale(scale, g, bias, fMax);
}

}; // namespace ibse
