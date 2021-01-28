#include <ibse_internal.h>
#include <ibse_testing.h>


int
main()
{
   // input
   Eigen::Matrix3d F;
   F << 0,1,0,  0,0,1,  0,0,0;
   Eigen::Vector3d L(0,0,1);
   double qc = 55;
   double dt = 1/30.0;

   // output
   Eigen::Matrix3d A;
   A << 0,0,0,0,0,0,0,0,0;
   Eigen::Matrix3d Q;
   Q << 0,0,0,0,0,0,0,0,0;

   ibse::ltiDisc(F, L, qc, dt, A, Q);

   // Should be:
   // A = 1.0000    0.0333    0.0006
   //          0    1.0000    0.0333
   //          0         0    1.0000
   //
   // Q = 0.0000    0.0000    0.0003
   //     0.0000    0.0007    0.0306
   //     0.0003    0.0306    1.8342

   APX_EQ(A(0,0), 1.0000); APX_EQ(A(0,1), 0.0333); APX_EQ(A(0,2), 0.00055);
   APX_EQ(A(1,0), 0.0000); APX_EQ(A(1,1), 1.0000); APX_EQ(A(1,2), 0.0333);
   APX_EQ(A(2,0), 0.0000); APX_EQ(A(2,1), 0.0000); APX_EQ(A(2,2), 1.0000);

   APX_EQ(Q(0,0), 0.0000); APX_EQ(Q(0,1), 0.0000); APX_EQ(Q(0,2), 0.0003);
   APX_EQ(Q(1,0), 0.0000); APX_EQ(Q(1,1), 0.0007); APX_EQ(Q(1,2), 0.0306);
   APX_EQ(Q(2,0), 0.0003); APX_EQ(Q(2,1), 0.0306); APX_EQ(Q(2,2), 1.8333);
}


