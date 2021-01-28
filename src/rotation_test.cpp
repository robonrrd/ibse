#include <ibse_testing.h>
#include <ibse_internal.h>
#include <vector>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <stdlib.h>



int
main()
{
   const size_t N = 10;

   Eigen::ArrayXXd quats(N,4);
   Eigen::ArrayXXd pts(N,3);

   for (size_t ii=0; ii<N; ++ii)
   {
      // Random rotation
      Eigen::Quaterniond qq(drand48(), 0.5-drand48(), 0.5-drand48(), 0.5-drand48());
      qq.normalize();
      quats.row(ii) << qq.w(), qq.x(), qq.y(), qq.z();

      pts.row(ii) << 0.5-drand48(), 0.5-drand48(), 0.5-drand48();
   }

   
   Eigen::ArrayXXd out = ibse::rotatePoints(quats, pts);  

   std::cout << "quats: " << std::endl << quats << std::endl;
   std::cout << "input: " << std::endl << pts << std::endl;
   std::cout << "output: " << std::endl << out << std::endl;
}
