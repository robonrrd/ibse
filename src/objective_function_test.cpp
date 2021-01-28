#include <iostream>
#include <stdlib.h>
#include <ibse_testing.h>
#include <ibse_internal.h>


int
main()
{
   const size_t iterations = 10;
   const size_t dim = 7;
   const size_t N   = 10;
   
   for (size_t itr=0; itr<iterations; ++itr)
   {
      // Set random data
      const int szFreqRange = 1;

      Eigen::MatrixXd A(dim, N);
      for (size_t ii=0; ii<dim; ++ii)
         for (size_t jj=0; jj<N; ++jj)
            A(ii, jj) = 1.0 - 2.0*drand48();
   
      Eigen::MatrixXd b(dim, 1);
      for (size_t ii=0; ii<dim; ++ii)
         b(ii, 0) = 1.0 - 2.0*drand48();

      // Precompute the FFT of 'b'
      const size_t Nb = b.rows() / 3;
      cv::Mat Ai(3, Nb, CV_64F);
      for (size_t ii=0; ii<Nb; ++ii)
         for (size_t jj=0; jj<3; ++jj)
            Ai.at<double>(jj, ii) = b(ii*3+jj,0);
      
      cv::Mat Fi(3, Nb, CV_64F);
      cv::dft(Ai, Fi, cv::DFT_ROWS);
      cv::Mat Fit(3, szFreqRange, CV_64F);
      Fit = cv::abs(Fi(cv::Range(0, 3), cv::Range(0, szFreqRange)));
      
      double* x = new double[dim];
      for (size_t ii=0; ii<dim; ++ii)
         x[ii] = 1.0 - 2.0*drand48();

      ibse::OptimizationData data;
      data.A = &A;
      data.Fi = &Fit;
      data.szFreqRange = szFreqRange;
      data.numIters = 0;

      double eigen_obj = ibse::freqDomainObjective(dim, x, 0, (void*)&data);
      double opencv_obj = ibse::opencv_freqDomainObjective(dim, x, 0, (void*)&data);

      APX_EQ(eigen_obj, opencv_obj);
   }
}; 



