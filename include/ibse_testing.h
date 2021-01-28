#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#define APX_EQ_EPS(X, Y, EPS) assert(fabs(X-Y) < EPS)

static const float eps = 1e-4;
#define APX_EQ(X, Y) assert(fabs(X-Y) < eps)

#define SIZE(X) std::cout << #X << " " << X.rows() << "x" << X.cols() << std::endl;


// Reading data from text files
std::vector<Eigen::Vector3d> readPosition(const std::string &fn);
std::vector<double> readTime(const std::string &fn);
