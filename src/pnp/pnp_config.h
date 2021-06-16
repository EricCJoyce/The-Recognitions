#ifndef __PNP_CONFIG_H
#define __PNP_CONFIG_H

#include <dirent.h>
#include <iostream>
#include <limits>                                                   //  Needed for int-infinity
#include <opencv2/calib3d.hpp>                                      //  Needed to identify PnP-solver methods like SOLVEPNP_AP3P
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "descriptor.h"                                             //  Needed for _BRISK, _ORB, _SIFT, _SURF... #definitions

#define _BOX_BLUR                0
#define _GAUSSIAN_BLUR           1
#define _MEDIAN_BLUR             2

#define MINIMUM_RATIO_THRESHOLD 0.001                               /* Doesn't make sense to admit 0.0: we would never match anything. */

#define _MATCH_MODE_SIG2Q        0                                  /* Train on Signatures and match to Query */
#define _MATCH_MODE_Q2SIG        1                                  /* Train on Query and match to Signatures */
#define _MATCH_MODE_MUTUAL       2                                  /* Train on both and match mutually nearest neighbors */

#define _POSE_EST_BY_VOTE        0                                  /* Use only correspondences that made up an object's votes */
#define _POSE_EST_INDEPENDENT    1                                  /* Make a fresh survey of correspondences for each object considered */

#define PARAM_NONE               0                                  /* Flag indicating that no argument is to follow */
#define PARAM_Q                  1                                  /* Flag indicating that a query image file path is to follow */
#define PARAM_SIG                2                                  /* Flag indicating that a signature file or directory is to follow */
#define PARAM_K                  3                                  /* Flag indicating that an intrinsics file name is to follow */
#define PARAM_RT                 4                                  /* Flag indicating that an extrinsics file name is to follow */
#define PARAM_KDTREE             5                                  /* Flag indicating that the number of KDtrees in index is to follow */
#define PARAM_HASHTABLES         6                                  /* Flag indicating that the number of LSH tables in index is to follow */
#define PARAM_DISTORT            7                                  /* Flag indicating that a distortion coefficients file name is to follow */
#define PARAM_ITER               8                                  /* Flag indicating that RANSAC's iteration count is to follow */
#define PARAM_REPROJ_ERR         9                                  /* Flag indicating that RANSAC's reprojection error is to follow */
#define PARAM_CONF              10                                  /* Flag indicating that RANSAC's pose confidence goal is to follow */
#define PARAM_METHOD            11                                  /* Flag indicating that RANSAC's method is to follow */
#define PARAM_DOWNSAMPLE        12                                  /* Flag indicating that the downsampling scalar is to follow */
#define PARAM_BLUR_TYPE         13                                  /* Flag indicating which blur method to use on the query image */
#define PARAM_BLUR_K_SIZE       14                                  /* Flag indicating the size of the blur kernel to use */
#define PARAM_TOP_K             15                                  /* Flag indicating that the number of top candidates is to follow */
#define PARAM_RATIO_THRESH      16                                  /* Flag indicating that the ratio threshold FOR BOTH SIG-->Q and Q-->SIG is to follow */
#define PARAM_RATIO_SIG2Q       17                                  /* Flag indicating that the ratio threshold FOR SIG-->Q ONLY is to follow */
#define PARAM_RATIO_Q2SIG       18                                  /* Flag indicating that the ratio threshold FOR Q-->SIG ONLY is to follow */
#define PARAM_MAX_NN_L2_DIST    19                                  /* Flag indicating the maximum L2 distance for nearest neighbors is to follow */
#define PARAM_MAX_NN_HAMM_DIST  20                                  /* Flag indicating the maximum Hamming distance for nearest neighbors is to follow */
#define PARAM_POSE_EST_METHOD   21                                  /* Flag indicating that the argument to follow sets the pose estimation method */
#define PARAM_Q_LIMIT_FEATURES  22                                  /* Flag indicating that the argument to follow sets the number of query-features to keep */
#define PARAM_BRISK_CONFIG      23                                  /* Flag indicating that a BRISK configuration file path is to follow */
#define PARAM_ORB_CONFIG        24                                  /* Flag indicating that an ORB configuration file path is to follow */
#define PARAM_SIFT_CONFIG       25                                  /* Flag indicating that a SIFT configuration file path is to follow */
#define PARAM_SURF_CONFIG       26                                  /* Flag indicating that a SURF configuration file path is to follow */
#define PARAM_MASK              27                                  /* Flag indicating that an extractor mask file path is to follow */
#define PARAM_NONMAXSUPPR       28                                  /* Flag indicating that a non-maximum suppression parameter is to follow */
#define PARAM_MATCH_MODE        29                                  /* Flag indicating that a matching-mode string is to follow */

/*
#define __PNP_CONFIG_DEBUG 1
*/

using namespace cv;
using namespace std;

/**************************************************************************************************
 PnPConfig  */
class PnPConfig
  {
    public:
      PnPConfig();                                                  //  No information provided (use defaults)
      PnPConfig(int, char**);                                       //  Construct using arguments
      ~PnPConfig();                                                 //  Destructor

      unsigned int fetchSignatures(char**, unsigned int**) const;   //  Write signature file names, file name lengths; return signature count

      bool parse(int, char**);

      unsigned int Sig(char**) const;                               //  Copy the signatures' file path into the given buffer
      unsigned int Q(char**) const;                                 //  Copy the query image into the given buffer
      unsigned int K(char**) const;                                 //  Copy the name of the K file into the given buffer
      unsigned int Rt(char**) const;                                //  Copy the name of the Rt file into the given buffer
      unsigned int distortion(char**) const;                        //  Copy the name of the distortion file into the given buffer
      unsigned int mask(char**) const;                              //  Copy the name of the mask file into the given buffer
      bool hasMask() const;

      unsigned int iterations() const;
      float reprojectionError() const;
      float confidence() const;
      int ransacMethod() const;
      unsigned char minimalRequired() const;
      unsigned int qLimit() const;

      unsigned char blurMethod() const;
      unsigned char blurKernelSize() const;
      float downSample() const;
      unsigned int numKDtrees() const;
      unsigned int numTables() const;
      unsigned int topK() const;
      unsigned char matchMethod() const;
      float ratioThresholdSig2Q() const;
      float ratioThresholdQ2Sig() const;
      float maximumNNL2Dist() const;
      unsigned int maximumNNHammingDist() const;
      unsigned char poseEstMethod() const;
      bool refineEstimate() const;
      bool convertToGrayscale() const;
      float nonMaxSuppression() const;

      int brisk_threshold() const;
      int brisk_octaves() const;
      float brisk_patternScale() const;

      float orb_scaleFactor() const;
      int orb_levels() const;
      int orb_edgeThreshold() const;
      int orb_firstLevel() const;
      int orb_wta_k() const;
      int orb_scoreType() const;
      int orb_patchSize() const;
      int orb_fastThreshold() const;

      int sift_octaveLayers() const;
      double sift_contrastThreshold() const;
      double sift_edgeThreshold() const;
      double sift_sigma() const;

      double surf_hessianThreshold() const;
      int surf_octaves() const;
      int surf_octaveLayers() const;

      bool renderBlurred() const;                                   //  Are we rendering the blurred query image? (Handled by Extractor)
      bool renderFeatures() const;                                  //  Are we rendering the detected features? (Handled by Extractor)
      bool renderInliers() const;                                   //  Are we rendering the inliers? (Handled by ./pnp)
      unsigned char renderInliersMode() const;                      //  HOW are we rendering the inliers? (Handled by ./pnp)
      bool writeDetected() const;                                   //  Are we writing (X, Y) detected in query image to file? (Handled by Extractor)
      bool writeObjCorr() const;                                    //  Are we writing each object's correspondences to text file? (Handled by ./pnp)
      bool writeObjInliers() const;                                 //  Are we writing each object's inliers to text file? (Handled by ./pnp)

      bool verbose() const;                                         //  How chatty is the program?
      bool helpme() const;                                          //  If true, do not run the program at all: print the usage page.

      void display() const;
      void paramUsage() const;                                      //  Display parameter usage text

    private:
      char* QFP;                                                    //  Query image filepath
      unsigned int QFPlen;

      ////////////////////////////////////////////////////////////////  Signatures
      char* SigFP;                                                  //  Filepath for the signature(s)
      unsigned int SigFPlen;
      bool SigFPIsDirectory;

      ////////////////////////////////////////////////////////////////  Linear algebraic entities, K, Rt, r, t, ...
      char* KFN;                                                    //  Intrinsic matrix file name
      unsigned int KFNlen;                                          //  Length of that string
      char* RtFN;                                                   //  Extrinsic matrix file name
      unsigned int RtFNlen;                                         //  Length of that string
      char* distortionFN;                                           //  Distortion coefficients file name
      unsigned int distortionFNlen;                                 //  Length of that string

      ////////////////////////////////////////////////////////////////  Mask file path
      char* maskFP;
      unsigned int maskFPlen;

      ////////////////////////////////////////////////////////////////  RANSAC parameters
      unsigned int _iterations;                                     //  The number of iterations to run RANSAC
      float _reprojectionError;                                     //  The tolerable reprojection error
      float _confidence;                                            //  Halt RANSAC if we are this confident about our pose
      int _ransacMethod;                                            //  The RANSAC method
      unsigned char _minimalRequired;                               //  Least number of correspondences needed to solve PnP for a given method
      unsigned int _qFeaturesLimit;                                 //  Limit on the number of query features

      ////////////////////////////////////////////////////////////////  Image blur
      unsigned char _blurMethod;
      unsigned char _blurKernelSize;

      ////////////////////////////////////////////////////////////////  Image processing
      float _downSample;
      bool _convertToGrayscale;

      ////////////////////////////////////////////////////////////////  Program parameters
      unsigned int _numKDtrees;                                     //  Number of KD-trees to use in our Index
      unsigned int _numTables;                                      //  Number of hash tables (OpenCV p. 578)
      unsigned int _topK;                                           //  Number of top object candidates to consider for pose estimation
      float _ratioThreshold_Sig2Q;                                  //  Lowe's ratio test, Sig-->Q
      float _ratioThreshold_Q2Sig;                                  //                     Q-->Sig
      float _maximumNNL2Dist;                                       //  In order to be nearest neighbors in any meaningful sense, require that they
      unsigned int _maximumNNHammingDist;                           //  be at most this "far" apart.
      unsigned char _poseEstMethod;                                 //  Which set of correspondences to use in pose estimation
      bool _refineEstimate;                                         //  Whether to re-run RANSAC using only the best pose's inliers
      unsigned char _matchMode;                                     //  Sig-->Q, Q-->Sig, Sig<-->Q

      ////////////////////////////////////////////////////////////////  Detector tuning
      float _nonMaxSuppression;                                     //  The radius in pixels around a feature within which to non-max suppress

      char* briskConfigFP;                                          //  BRISK configuration file path
      unsigned int briskConfigFPlen;
      int _brisk_threshold;                                         //  BRISK: AGAST detection threshold
      int _brisk_octaves;                                           //  BRISK: Detection octaves; 0 = single-scale
      float _brisk_patternScale;                                    //  BRISK: Scale applied to pattern used for sampling keypoint neighborhood

      char* orbConfigFP;                                            //  ORB configuration file path
      unsigned int orbConfigFPlen;
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

      char* siftConfigFP;                                           //  SIFT configuration file path
      unsigned int siftConfigFPlen;
      int _sift_octaveLayers;                                       //  SIFT: Number of layers in each octave.
      double _sift_contrastThreshold;                               //  SIFT: Larger threshold --> FEWER features produced by the detector
      double _sift_edgeThreshold;                                   //  SIFT: Larger threshold --> MORE features are retained
      double _sift_sigma;                                           //  SIFT: Sigma of Gaussian filter applied to input image at octave[0].
                                                                    //        If your input image was captured by a weak camera with a soft lens, you might
                                                                    //        want to LOWER this number.

      char* surfConfigFP;                                           //  SURF configuration file path
      unsigned int surfConfigFPlen;
      double _surf_hessianThreshold;                                //  SURF: Threshold for Hessian key point detector
      int _surf_octaves;                                            //  SURF: Number of pyramid octaves the detector will use
      int _surf_octaveLayers;                                       //  SURF: Number of layers within each octave

      ////////////////////////////////////////////////////////////////  Auxiliary Output flags
      bool _renderBlurred;                                          //  Whether to write a copy of the blurred query image to file
      bool _renderFeatures;                                         //  Whether to write an image showing the detected query image features
      bool _renderInliers;                                          //  Whether to write an image showing the inliers in the query image
      bool _writeDetected;                                          //  Whether to write detected features (X, Y) to (text) file
      bool _writeObjCorr;                                           //  Whether to write object correspondences to (text) file
      bool _writeObjInliers;                                        //  Whether to write object inliers to (text) file

      bool _verbose;                                                //  Chatty or shut up?
      bool _helpme;

      bool isSignaturePathDirectory() const;

      bool loadBRISKparams();                                       //  Read the given binary file and set BRISK parameters accordingly
      bool loadORBparams();                                         //  Read the given binary file and set ORB parameters accordingly
      bool loadSIFTparams();                                        //  Read the given binary file and set SIFT parameters accordingly
      bool loadSURFparams();                                        //  Read the given binary file and set SURF parameters accordingly
  };

#endif