/*  Eric C. Joyce, Stevens Institute of Technology, 2020.

    Class for the Query Feature-Extractor.
*/

#ifndef __EXTRACTOR_H
#define __EXTRACTOR_H

#include <iostream>
#include <limits>                                                   //  Needed for int-infinity
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>                                      //  Needed for blur
#include <opencv2/opencv.hpp>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                                                 //  Do I need this?
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "descriptor.h"
#include "pnp_config.h"

#define _BOX_BLUR       0
#define _GAUSSIAN_BLUR  1
#define _MEDIAN_BLUR    2

#define _BRISK_FLAG     1
#define _ORB_FLAG       2
#define _SIFT_FLAG      4
#define _SURF_FLAG      8

#define _ORB_DETECTION_DEFAULT_MAXIMUM  1000000                     /* Had to pick somthing because for whatever reason I can't pass
                                                                       numeric_limits<unsigned int>::max() to the ORB constructor.
                                                                       (I mean, I *can*, but it find zero features. Of course.) */

/*
#define __EXTRACTOR_DEBUG 1
*/

using namespace cv;
using namespace std;

/**************************************************************************************************
 Extractor  */
class Extractor
  {
    public:
      Extractor();                                                  //  Constructors
      Extractor(unsigned char);
      ~Extractor();                                                 //  Destructor

      void initDetectors();
      unsigned int extract(cv::Mat);                                //  Extract features
      unsigned int extract(cv::Mat, cv::Mat);                       //  Apply a mask to extraction
      void descMat(unsigned char, cv::Mat*) const;                  //  Return a matrix of all features of given type
      unsigned int features() const;                                //  Return the number of features detected
      unsigned int features(unsigned char) const;                   //  Return the number of features detected of given type

      void setFlags(unsigned char);                                 //  Set detector flags according to a bit-array
      void setBRISK(bool);                                          //  Specifically toggle use of BRISK
      void setORB(bool);                                            //  Specifically toggle use of ORB
      void setSIFT(bool);                                           //  Specifically toggle use of SIFT
      void setSURF(bool);                                           //  Specifically toggle use of SURF

      void setFeatureLimit(unsigned int);

      void setBRISKparams(PnPConfig*);                              //  Set BRISK-detector parameters
      void setORBparams(PnPConfig*);                                //  Set ORB-detector parameters
      void setSIFTparams(PnPConfig*);                               //  Set SIFT-detector parameters
      void setSURFparams(PnPConfig*);                               //  Set SURF-detector parameters

      void disableBlur();
      void setBlurMethod(unsigned char);
      void setBlurKernelSize(unsigned char);
      void disableDownSample();
      void setDownSample(float);
      void setRenderBlur(bool);                                     //  Toggle whether we render the blurred image
      void setRenderDetections(bool);                               //  Toggle whether we render the detected interest points

      unsigned int featureLimit() const;
      bool BRISK() const;                                           //  Return whether BRISK is in use
      bool ORB() const;                                             //  Return whether ORB is in use
      bool SIFT() const;                                            //  Return whether SIFT is in use
      bool SURF() const;                                            //  Return whether SURF is in use

      unsigned char type(unsigned int) const;                       //  Return the i-th feature's type
      float x(unsigned int) const;                                  //  Return the i-th feature's X coordinate
      float y(unsigned int) const;                                  //  Return the i-th feature's Y coordinate
      float size(unsigned int) const;                               //  Return the i-th feature's size
      float angle(unsigned int) const;                              //  Return the i-th feature's angle
      float response(unsigned int) const;                           //  Return the i-th feature's response
      signed int octave(unsigned int) const;                        //  Return the i-th feature's octave
      unsigned char vec(unsigned int, void**) const;                //  Copy descriptor vector into given (uchar or float) buffer

      void reset();                                                 //  Clear all keypoint and descriptor vectors

    private:
      ////////////////////////////////////////////////////////////////  Detectors
      Ptr<FeatureDetector> brisk_detect;                            //  BRISK, if we're using BRISK
      Ptr<FeatureDetector> orb_detect;                              //  ORB, if we're using ORB
      Ptr<cv::xfeatures2d::SIFT> sift_detect;                       //  SIFT, if we're using SIFT
      Ptr<cv::xfeatures2d::SURF> surf_detect;                       //  SURF, if we're using SURF
      unsigned int limitFeatures;                                   //  By default, impose no limits on Q features

      ////////////////////////////////////////////////////////////////  Detector tuning
      int _brisk_threshold;                                         //  BRISK: AGAST detection threshold
      int _brisk_octaves;                                           //  BRISK: Detection octaves; 0 = single-scale
      float _brisk_patternScale;                                    //  BRISK: Scale applied to pattern used for sampling keypoint neighborhood

      float _orb_scaleFactor;                                       //  ORB: Pyramid decimation ratio greater than 1.
                                                                    //       scaleFactor == 2 means classical pyramid where each layer has 4x fewer pixels
      int _orb_levels;                                              //  ORB: Number of pyramid levels
      int _orb_edgeThreshold;                                       //  ORB: Size of the border at which features are not detected.
                                                                    //       Should roughly equal patch size
      int _orb_firstLevel;                                          //  ORB: The level of the pyramid onto which we put the source image.
                                                                    //       Any previous layers are filled with upscaled versions of the source.
      int _orb_wta_k;                                               //  ORB: Number of points that produce each element of the oriented BRIEF descriptor.
                                                                    //       Default value 2 means taking a random pair of points and comparing their
                                                                    //       brightnesses for a {0, 1} response. Other values are 3 and 4.
      int _orb_scoreType;                                           //  ORB: Default value ORB::HARRIS_SCORE means that the Harris algo ranks features.
                                                                    //       An alternative is ORB::FAST_SCORE (faster, but a bit less stable).
      int _orb_patchSize;                                           //  ORB: Size of the patch used by the oriented BRIEF descriptor. On smaller pyramid
                                                                    //       layers the perceived image area covered by a feature will be larger.
      int _orb_fastThreshold;                                       //  ORB: The fast threshold

      int _sift_octaveLayers;                                       //  SIFT: Number of layers in each octave.
      double _sift_contrastThreshold;                               //  SIFT: Larger threshold --> FEWER features produced by the detector
      double _sift_edgeThreshold;                                   //  SIFT: Larger threshold --> MORE features are retained
      double _sift_sigma;                                           //  SIFT: Sigma of Gaussian filter applied to input image at octave[0].
                                                                    //        If your input image was captured by a weak camera with a soft lens, you might
                                                                    //        want to LOWER this number.

      double _surf_hessianThreshold;                                //  SURF: Threshold for Hessian key point detector
      int _surf_octaves;                                            //  SURF: Number of pyramid octaves the detector will use
      int _surf_octaveLayers;                                       //  SURF: Number of layers within each octave

      ////////////////////////////////////////////////////////////////  Detector flags
      bool useBRISK;
      bool useORB;
      bool useSIFT;
      bool useSURF;

      ////////////////////////////////////////////////////////////////  Query image key-points and descriptors
      Descriptor** d;                                               //  Length N for N features (of all types) in the query image
      unsigned int* descriptorCtr;                                  //  Counts of each type of descriptor

      ////////////////////////////////////////////////////////////////  Image blur
      unsigned char blurMethod;
      unsigned char blurKernelSize;

      ////////////////////////////////////////////////////////////////  Down-sampling
      float downSample;

      ////////////////////////////////////////////////////////////////  Debugging and diagnostic utilities
      bool renderBlurred;                                           //  Whether to render the blurred image, just to see it
      bool renderDetections;                                        //  Whether to render the detected features, just to see them
  };

#endif