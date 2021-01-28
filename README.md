# Inertia-Based Scale Estimation


A fast C++ implementation of the IROS 2017 paper “Inertial-Based Scale Estimation for Structure from Motion on Mobile Devices”  https://arxiv.org/abs/1611.09498


## Motivation

We can do SfM on a smart phone or tablet, but with only one camera you will not have metric scale.

We can recover the metric scale using IMU (gyroscope and linear accelerometer) information.
However, the visual data (images) and the IMU readings may not be synchronized in time.

We have an unknown rotation between camera and IMU frames of reference, as well as an unknown delta-T

Compare gyroscope and visual angular velocity: simple optimization problem.

Find scaling, gravity vector and accelerometer bias through another optimization problem.

In reality, visual estimates of velocity or acceleration are noisy (motion blur, rolling shutter)

Kalman filter and Rauch-Tung-Striebel smoother to smooth visual acceleration.

Perform scale estimation in frequency domain, rather than time domain: this helps with noisy IMU and is not sensitive to the delta-T which can vary during SfM process.


## Results
Avg error: <3% after 2 meters of camera travel.  <1% after 14m

