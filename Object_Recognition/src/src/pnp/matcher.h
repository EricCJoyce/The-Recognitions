/*  Eric C. Joyce, Stevens Institute of Technology, 2020.

    Class for feature matching.
*/

#ifndef __MATCHER_H
#define __MATCHER_H

#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "descriptor.h"
#include "extractor.h"
#include "pnp_config.h"
#include "signature.h"

/*
#define __MATCHER_DEBUG 1
*/

using namespace cv;
using namespace std;

/**************************************************************************************************
 Typedefs  */

typedef struct MatchType
  {
    unsigned char type;                                             //  #defined in descriptor.h
                                                                    //  "The 'sigFeature'-th feature of 'type'
                                                                    //  in the 'signature'-th object matches
                                                                    //  the 'queryFeature'-th feature (also of 'type')
                                                                    //  in the query image."
    signed int signature;                                           //  Match with Signature.index()
    unsigned int sigFeature;                                        //  The i-th Descriptor of this 'type' in 'signature'
    unsigned int queryFeature;                                      //  The j-th Descriptor of this 'type' in Query
  } Match;

/**************************************************************************************************
 Matcher  */
class Matcher
  {
    public:
      Matcher();                                                    //  Constructors
      Matcher(PnPConfig);
      ~Matcher();                                                   //  Destructor

      unsigned int train(Signature**, unsigned int);                //  Take an array of Signatures, extract-concat matrices, and train indices
      unsigned int train(Extractor*);                               //  Take the Extractor's features as matrices, train indices

      unsigned int match(Extractor*);                               //  Match the Extractor's features against the "trained" Signatures
      unsigned int match(Signature**, unsigned int);                //  Match Signatures against the "trained" query image features
      unsigned int match(void);                                     //  Match mutually

      unsigned int top(unsigned int**, unsigned int**) const;
      unsigned int correspondences(Signature*, Extractor*, std::vector<cv::Point2f>*, std::vector<cv::Point3f>*) const;

      unsigned int numKDtrees(void) const;
      unsigned int numTables(void) const;
      unsigned int topK(void) const;
      float ratio_Sig2Q(void) const;
      float ratio_Q2Sig(void) const;
      float maxL2(void) const;
      unsigned int maxHamming(void) const;

      void clearSig(void);                                          //  Reset the Signature indices only
      void clearQ(void);                                            //  Reset the Query indices only
      void clear(void);                                             //  Reset all indices

      void printSignatureFeatures(void) const;

    private:
      ////////////////////////////////////////////////////////////////  General behavior parameters
      unsigned int _numKDtrees;                                     //  Number of KD-trees used for L2-distance matching
      unsigned int _numTables;                                      //  Number of hash tables used for Hamming-distance matching
      unsigned int _topK;                                           //  Number of top object candidates to consider for pose estimation
      float _ratioThreshold_Sig2Q;                                  //  Lowe's ratio test
      float _ratioThreshold_Q2Sig;
      float _maximumNNL2Dist;                                       //  In order to be nearest neighbors in any meaningful sense, require that they
      unsigned int _maximumNNHammingDist;                           //  be at most this "far" apart.

      ////////////////////////////////////////////////////////////////  Track features from SIGNATURES
      unsigned int numSignatures;                                   //  How many objects' features have we?

      unsigned int* BRISK_signatureLookup;                          //  These are all arrays of Signature indices:
      unsigned int* ORB_signatureLookup;                            //  SIFT_signatureLookup[i] = j means that the i-th SIFT vector
      unsigned int* SIFT_signatureLookup;                           //  belongs to the j-th Signature.
      unsigned int* SURF_signatureLookup;

      unsigned int* BRISK_signatureFeature;                         //  These are all arrays of feature indices:
      unsigned int* ORB_signatureFeature;                           //  SIFT_feature_map[i] = j means that the i-th SIFT vector
      unsigned int* SIFT_signatureFeature;                          //  (of all SIFT vectors) is the j-th one for that object.
      unsigned int* SURF_signatureFeature;

      unsigned int BRISK_sigLen;                                    //  Length of both BRISK_signatureLookup and BRISK_signatureFeature
      unsigned int ORB_sigLen;                                      //  Length of both ORB_signatureLookup and ORB_signatureFeature
      unsigned int SIFT_sigLen;                                     //  Length of both SIFT_signatureLookup and SIFT_signatureFeature
      unsigned int SURF_sigLen;                                     //  Length of both SURF_signatureLookup and SURF_signatureFeature

      cv::FlannBasedMatcher BRISK_SigIndex;                         //  Matcher trained on Signature features; BRISK uses HAMMING distances
      cv::FlannBasedMatcher ORB_SigIndex;                           //  Matcher trained on Signature features; ORB uses HAMMING distances
      cv::FlannBasedMatcher SIFT_SigIndex;                          //  Matcher trained on Signature features; SIFT uses L2 distances
      cv::FlannBasedMatcher SURF_SigIndex;                          //  Matcher trained on Signature features; SURF uses L2 distances

      ////////////////////////////////////////////////////////////////  Track features from THE QUERY IMAGE
      unsigned int BRISK_QLen;                                      //  Number of BRISK features extracted from the query (used to train BRISK_QIndex)
      unsigned int ORB_QLen;                                        //  Number of ORB features extracted from the query (used to train ORB_QIndex)
      unsigned int SIFT_QLen;                                       //  Number of SIFT features extracted from the query (used to train SIFT_QIndex)
      unsigned int SURF_QLen;                                       //  Number of SURF features extracted from the query (used to train SURF_QIndex)

      cv::FlannBasedMatcher BRISK_QIndex;                           //  Matcher trained on Query features; BRISK uses HAMMING distances
      cv::FlannBasedMatcher ORB_QIndex;                             //  Matcher trained on Query features; ORB uses HAMMING distances
      cv::FlannBasedMatcher SIFT_QIndex;                            //  Matcher trained on Query features; SIFT uses L2 distances
      cv::FlannBasedMatcher SURF_QIndex;                            //  Matcher trained on Query features; SURF uses L2 distances

      ////////////////////////////////////////////////////////////////  However determined, the matches:
      Match* matches;
      unsigned int numMatches;

      void quicksort(unsigned int**, unsigned int**, unsigned int, unsigned int) const;
      unsigned int partition(bool desc, unsigned int**, unsigned int**, unsigned int, unsigned int) const;
  };

#endif