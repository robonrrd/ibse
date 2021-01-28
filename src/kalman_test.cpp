#include <ibse_testing.h>
#include <ibse_internal.h>
#include <iostream>

int
main()
{
   const std::string data_path = "../data/";
   const std::string pose_file = "kalman_test_poses.txt";

   // read visual data
   std::cout << "Reading visual data.." << std::flush;
   std::vector<Eigen::Vector3d> posVis = readPosition(data_path+pose_file);
   std::vector<double> tVis = readTime(data_path+pose_file);
   std::cout << " done" << std::endl;   

   double dt = 0;
   for (size_t ii=1; ii<tVis.size(); ++ii)
      dt += (tVis[ii] - tVis[ii-1]);
   dt = dt / (tVis.size()-1);

   Eigen::ArrayXXd A = ibse::KalmanRTS(posVis, dt);

   // Should be:
   //
   // -0.662419945981779   1.824101295842961   2.896776974020415
   // -0.662431489522616   1.824321166807774   2.897104972439537
   // -0.661824708158484   1.823812717089266   2.896018841572308
   // -0.659867235362322   1.820807777017613   2.890165512356016
   // -0.656868141629438   1.815861715760839   2.879995505209227
   // -0.653567029120227   1.812221827277211   2.868579315074893
   // -0.650734783670111   1.810821083174493   2.860134199320156
   // -0.648676785506472   1.810427285247322   2.855654089336448
   // -0.647659382042185   1.810439295623441   2.854072564870451
   // -0.647493579432640   1.810456828478497   2.853854841143328

   APX_EQ(A(0,0), -0.66242); APX_EQ(A(1,0), 1.82410); APX_EQ(A(2,0), 2.89678);
   APX_EQ(A(0,1), -0.66243); APX_EQ(A(1,1), 1.82432); APX_EQ(A(2,1), 2.89710);
   APX_EQ(A(0,2), -0.66182); APX_EQ(A(1,2), 1.82381); APX_EQ(A(2,2), 2.89601);
   APX_EQ(A(0,3), -0.65986); APX_EQ(A(1,3), 1.82080); APX_EQ(A(2,3), 2.89016);
   APX_EQ(A(0,4), -0.65686); APX_EQ(A(1,4), 1.81586); APX_EQ(A(2,4), 2.87999);
   APX_EQ(A(0,5), -0.65356); APX_EQ(A(1,5), 1.81222); APX_EQ(A(2,5), 2.86857);
   APX_EQ(A(0,6), -0.65073); APX_EQ(A(1,6), 1.81082); APX_EQ(A(2,6), 2.86013);
   APX_EQ(A(0,7), -0.64867); APX_EQ(A(1,7), 1.81042); APX_EQ(A(2,7), 2.85565);
   APX_EQ(A(0,8), -0.64765); APX_EQ(A(1,8), 1.81043); APX_EQ(A(2,8), 2.85407);
   APX_EQ(A(0,9), -0.64749); APX_EQ(A(1,9), 1.81045); APX_EQ(A(2,9), 2.85385);
}


