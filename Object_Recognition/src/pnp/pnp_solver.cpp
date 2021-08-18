#ifndef __PNP_SOLVER_CPP
#define __PNP_SOLVER_CPP

#include "pnp_solver.h"

/**************************************************************************************************
 Constructors  */

/* PnPSolver constructor, no data given */
PnPSolver::PnPSolver(cv::Mat Kmat, cv::Mat distVec)
  {
    #ifdef __PNPSOLVER_DEBUG
    cout << "PnPSolver::PnPSolver()" << endl;
    #endif

    K = Kmat.clone();                                               //  Deep copy
    distortionCoeffs = distVec.clone();                             //  Deep copy

    RtGuess = false;

    _iterations = 1000;                                             //  Default
    _reprojectionError = 5.0;                                       //  Default
    _confidence = 0.99;                                             //  Default
    ransacMethod = SOLVEPNP_AP3P;                                   //  Default
    minimalRequired = lookupMinimalRequired(ransacMethod);          //  According to ransacMethod default
  }

/**************************************************************************************************
 Destructor  */

PnPSolver::~PnPSolver()
  {
  }

/**************************************************************************************************
 Solve PnP  */

/* Simply attempt to compute a pose estimate. */
unsigned int PnPSolver::solve(std::vector<cv::Point2f> pt2, std::vector<cv::Point3f> pt3)
  {
    bool pnpSuccess = false;

    _inliers.clear();

    if(pt3.size() >= minimalRequired)                               //  Can't estimate pose if there are not enough correspondences
      {
        if(RtGuess)                                                 //  If we have an initial guess, then apply it each time
          {
            r = rGiven.clone();
            t = tGiven.clone();
          }

        pnpSuccess = cv::solvePnPRansac(pt3,                        //  INPUT:  3D points from object
                                        pt2,                        //  INPUT:  2D points from query image
                                        K, distortionCoeffs,        //  INPUT:  Camera descriptions
                                        r,                          //  OUTPUT: 'r' will be in Rodrigues-form
                                        t,                          //  OUTPUT: *** THIS IS THE 4th COLUMN OF THE EXTRINSICS!!! ***
                                        RtGuess,                    //  INPUT:  whether to consider current values in 'r' and 't' as a guess
                                                                    //  INPUT:  command-line arguments or defaults
                                        _iterations, _reprojectionError, _confidence,
                                        _inliers,                   //  OUTPUT: yes, I'd like to know which were inliers (indices)
                                        ransacMethod);              //  INPUT:  {SOLVEPNP_P3P, SOLVEPNP_EPNP, SOLVEPNP_ITERATIVE, ... }
                                                                    //                                        ^ time-consuming
        if(pnpSuccess)                                              //  SUCCESS!
          return _inliers.size();
      }

    return 0;
  }

/* Copy r to the given cv::Mat */
void PnPSolver::rvec(cv::Mat* mat) const
  {
    if(r.cols > 0 && r.rows > 0)
      (*mat) = r.clone();
    return;
  }

/* Copy t to the given cv::Mat */
void PnPSolver::tvec(cv::Mat* mat) const
  {
    if(t.cols > 0 && t.rows > 0)
      (*mat) = t.clone();
    return;
  }

/**************************************************************************************************
 Setters  */

void PnPSolver::setK(cv::Mat Kmat)
  {
    K = Kmat.clone();
    return;
  }

void PnPSolver::setDist(cv::Mat distVec)
  {
    distortionCoeffs = distVec.clone();
    return;
  }

void PnPSolver::setInitialPose(cv::Mat rVec, cv::Mat tVec)
  {
    rGiven = rVec.clone();
    tGiven = tVec.clone();
    RtGuess = true;
    return;
  }

void PnPSolver::disableInitialPose(void)
  {
    RtGuess = false;
    return;
  }

void PnPSolver::setIterations(unsigned int i)
  {
    if(i > 0)
      _iterations = i;
    return;
  }

void PnPSolver::setReprojectionError(float e)
  {
    if(e >= 0.0)
      _reprojectionError = e;
    return;
  }

void PnPSolver::setConfidence(float c)
  {
    if(c < 1.0 && c >= 0.0)
      _confidence = c;
    return;
  }

void PnPSolver::setMethod(int m)
  {
    if(m == SOLVEPNP_ITERATIVE  ||
       m == SOLVEPNP_EPNP       ||
       m == SOLVEPNP_P3P        ||
       m == SOLVEPNP_DLS        ||
       m == SOLVEPNP_UPNP       ||
       m == SOLVEPNP_AP3P       ||
       m == SOLVEPNP_IPPE       ||
       m == SOLVEPNP_IPPE_SQUARE )
      ransacMethod = m;
    else                                                            //  Default to AP3P
      ransacMethod = SOLVEPNP_AP3P;
    minimalRequired = lookupMinimalRequired(ransacMethod);

    return;
  }

/**************************************************************************************************
 Getters  */

std::vector<int> PnPSolver::inliers() const
  {
    return _inliers;
  }

unsigned int PnPSolver::iterations() const
  {
    return _iterations;
  }

float PnPSolver::reprojectionError() const
  {
    return _reprojectionError;
  }

float PnPSolver::confidence() const
  {
    return _confidence;
  }

int PnPSolver::method() const
  {
    return ransacMethod;
  }

unsigned char PnPSolver::minimum() const
  {
    return minimalRequired;
  }

/**************************************************************************************************
 Utilities  */

unsigned char PnPSolver::lookupMinimalRequired(unsigned char method) const
  {
    if(method == SOLVEPNP_ITERATIVE)
      return 4;
    if(method == SOLVEPNP_EPNP)
      return 4;
    if(method == SOLVEPNP_P3P)
      return 4;
    if(method == SOLVEPNP_DLS)
      return 4;
    if(method == SOLVEPNP_UPNP)
      return 4;
    if(method == SOLVEPNP_AP3P)
      return 4;
    if(method == SOLVEPNP_IPPE)
      return 4;
    if(method == SOLVEPNP_IPPE_SQUARE)
      return 4;
    return 4;
  }

#endif