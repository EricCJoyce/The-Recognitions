/*  Eric C. Joyce, Stevens Institute of Technology, 2020.

    Class for the PnP-Solver.
*/

#ifndef __PNP_SOLVER_H
#define __PNP_SOLVER_H

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                                                 //  Do I need this?
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#define _RENDER_DESC_TYPE        0                                  /* Color inliers in the PLY according to their descriptor type */
#define _RENDER_COLORMAP         1                                  /* Color inliers in the PLY to match the color map of the 2D render */

/*
#define __PNPSOLVER_DEBUG 1
*/

using namespace cv;
using namespace std;

/**************************************************************************************************
 PnPSolver  */
class PnPSolver
  {
    public:
      PnPSolver(cv::Mat, cv::Mat);                                  //  Constructor
      ~PnPSolver();                                                 //  Destructor

                                                                    //  Run PnP-RANSAC to generate a pose estimate
      unsigned int solve(std::vector<cv::Point2f>, std::vector<cv::Point3f>);
      void rvec(cv::Mat*) const;                                    //  Copy 'r' to the given cv::Mat
      void tvec(cv::Mat*) const;                                    //  Copy 't' to the given cv::Mat

      void setK(cv::Mat);                                           //  (Re)Set the intrinsic matrix
      void setDist(cv::Mat);                                        //  (Re)Set the distortion coefficients' vector
      void setInitialPose(cv::Mat, cv::Mat);                        //  Provide an r and a t vector to use as an initial guess for pose
      void disableInitialPose(void);

      void setIterations(unsigned int);
      void setReprojectionError(float);
      void setConfidence(float);
      void setMethod(int);

      unsigned int iterations() const;
      float reprojectionError() const;
      float confidence() const;
      int method() const;
      unsigned char minimum() const;

      std::vector<int> inliers() const;

    private:
      cv::Mat K;                                                    //  The intrinsic matrix
      cv::Mat distortionCoeffs;                                     //  Vector of distortion coefficients

      cv::Mat rGiven;                                               //  If we have initial guesses, save them here and apply them to every estimate
      cv::Mat tGiven;
      bool RtGuess;                                                 //  Whether to consider initial values of 'r' and 't'
                                                                    //    as a guess for camera pose.

                                                                    //  Outputs are relative to the recognized object:
      cv::Mat r;                                                    //   * camera rotation, in Rodrigues-form
      cv::Mat t;                                                    //   * camera translation
      std::vector<int> _inliers;                                    //  Array of indices into 3D points and 2D points

                                                                    //  RANSAC parameters
      unsigned int _iterations;                                     //  The number of iterations to run RANSAC
      float _reprojectionError;                                     //  The tolerable reprojection error
      float _confidence;                                            //  Halt RANSAC if we are this confident about our pose
      int ransacMethod;                                             //  The RANSAC method
      unsigned char minimalRequired;                                //  Least number of correspondences needed to solve PnP for a given method
      bool doublePass;                                              //  Whether to run RANSAC again using only the inliers from the first estimate

      unsigned char lookupMinimalRequired(unsigned char) const;     //  Given a method, return the minimal number of correspondences
  };

#endif