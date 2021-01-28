#include <ibse_testing.h>
#include <ibse_internal.h>
#include <stdlib.h>
#include <iostream>

using namespace ibse;


Eigen::VectorXd 
linearInterpolation(const Eigen::VectorXd& x, const Eigen::VectorXd& y, Eigen::VectorXd& x_i)
{
   const size_t sz_x = x.size();
   const size_t sz_xi = x_i.size();
   Eigen::VectorXd y_i(sz_xi);

   size_t i0 = 0; 
   size_t i1 = 1;
   for (size_t ii=0; ii<sz_xi; ++ii)
   {
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


int
main()
{
   using namespace Eigen;

   const size_t N = 10;

   VectorXd x(N);
   for (size_t ii=0; ii<N; ++ii)
      x(ii) = ii*10;

   const double noise_amp = N*0.1;
   VectorXd y(N);
   for (size_t ii=0; ii<N; ++ii)
      y(ii) = ii*0.5 + (0.5*noise_amp - noise_amp*drand48());
   
   LinearSplineFunction interp(x, y);

   const size_t N_interp = N*10;
   VectorXd x_interp(N_interp);
   for (size_t ii=0; ii<N_interp; ++ii)
      x_interp(ii) = ii-N*0.5;

   VectorXd y_interp(N_interp);
   for (size_t ii=0; ii<N_interp; ++ii)
      y_interp(ii) = interp(x_interp[ii]);

   // std::cout << "x  : " << x.transpose() << std::endl;
   // std::cout << "y  : " << y.transpose() << std::endl;
   // std::cout << "y_i: " << y_interp.transpose() << std::endl;

   VectorXd y_interp2 = linearInterpolation(x, y, x_interp);
   // std::cout << "y_i2: " << y_interp2.transpose() << std::endl;

   // We know that the more expensive LinearSplineFunction is correct, so
   // we test against that
   for (size_t ii=0; ii<N_interp; ++ii)
      APX_EQ_EPS(y_interp(ii), y_interp2(ii), 1e-12);

}


