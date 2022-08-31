# Camera-Calibration-OpenCV-Python

---
# Source Code
Source code is compose of a single python class which you can import in your code. You can either feed already
calibrated camera matrix to undistort images or you can provide chessboard images with different orientation 
to generate camera matrix.

### Functionalities

- [x] generate calibration matrix
- [x] undistort a loaded frame
- [x] undistort images from directory
- [ ] undistort videos
- [ ] generate calibration matrix using video
- [ ] generate calibration matrix using webcam

## chessboard image
Currently, our source code use the chessboard mentioned below. Its dimension is (6,9), 
we take dimensions in terms of corners inside the chess board. To create your own chess board pattern,
please visit the following [link](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html).

![Screenshot](chessboard.png)

# Resources
Go through the below resourses in the following order in order to get better understanding regarding the
calibration. Althought the sections below cover each and everything in terms of OpenCv, but we would
recommend you to go through this amazing [playlist](https://www.youtube.com/watch?v=S-UHiFsn-GI&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo) 
by [Professor Shree K. Nayar](https://en.wikipedia.org/wiki/Shree_K._Nayar) on his YouTube channel.

### Understanding Geometric Image Formation
In geometric image formation we learn how the points in 3D space are related/projected to the 
points in 2D image plane. 

- [Video](https://www.youtube.com/watch?v=NIaICLR7D0Q)
- [BlogPost](https://learnopencv.com/geometry-of-image-formation/)

### Lens Distortion
As real world camera involve lenses, we need to get an understanding regarding lens effects on an image formation.

- [Video](https://www.youtube.com/watch?v=hzOeqCb2Fg4)
- [BlogPost](https://learnopencv.com/understanding-lens-distortion/)

### Camera Calibration
When using camera as visual sensor, it is important to know its parameters for effective usage.
Here are some [steps](https://stackoverflow.com/a/12821056) you could take to get accurate camera calibration.

- [BlogPost](https://learnopencv.com/camera-calibration-using-opencv/)
- [OpenCV-Official-Tutorial](https://docs.opencv.org/4.5.5/dc/dbb/tutorial_py_calibration.html)

# References

- [OpenCV](https://docs.opencv.org/4.5.5/)
- [LearnOpenCV](https://learnopencv.com/)
- [First Principles of Computer Vision](https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw)
