#ifndef __CAMERA_H
#define __CAMERA_H

#include <fstream>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAMERA_HEADER_LENGTH 512

/*
#define __CAMERA_DEBUG 1
*/

using namespace cv;
using namespace std;

/**************************************************************************************************
 Camera  */
class Camera
  {
    public:
      Camera();                                                     //  No information provided (use defaults)
      Camera(char*);                                                //  Construct using arguments
      ~Camera();                                                    //  Destructor

      bool load(char*);                                             //  Load from file

      void set_fx(float);                                           //  Setters
      void set_fy(float);
      void set_shear(float);
      void set_cx(float);
      void set_cy(float);

      void set_k1(float);
      void set_k2(float);
      void set_p1(float);
      void set_p2(float);
      void set_k3(float);
      void set_k4(float);
      void set_k5(float);
      void set_k6(float);

      void set_s1(float);
      void set_s2(float);
      void set_s3(float);
      void set_s4(float);

      void set_tx(float);
      void set_ty(float);

      float fx(void) const;                                         //  Getters
      float fy(void) const;
      float shear(void) const;
      float cx(void) const;
      float cy(void) const;

      float k1(void) const;
      float k2(void) const;
      float p1(void) const;
      float p2(void) const;
      float k3(void) const;
      float k4(void) const;
      float k5(void) const;
      float k6(void) const;

      float s1(void) const;
      float s2(void) const;
      float s3(void) const;
      float s4(void) const;

      float tx(void) const;
      float ty(void) const;

      void print(void) const;

      void K(cv::Mat*) const;
      void dist(cv::Mat*) const;

      unsigned int writeByteArray(char**) const;                    //  Convert an instance of this class to a byte array
      bool readByteArray(char*);                                    //  Update an instance of this class according to given byte array

    private:
      float _fx, _fy, _s, _cx, _cy;                                 //  Parts of intrinsic matrix
      float _k1, _k2;                                               //  Distortion vector
      float _p1, _p2;
      float _k3, _k4, _k5, _k6;
      float _s1, _s2, _s3, _s4;
      float _tx, _ty;
      char _header[CAMERA_HEADER_LENGTH];
  };

#endif
