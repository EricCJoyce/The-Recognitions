#ifndef __MATCHER_CPP
#define __MATCHER_CPP

#include "matcher.h"

/**************************************************************************************************
 Constructors  */

/* Matcher constructor, no data given */
Matcher::Matcher()
  {
    numSignatures = 0;                                              //  Initially, no Signatures

    _numKDtrees = 1;                                                //  Defaults
    _numTables = 10;
    _topK = 5;
    _ratioThreshold_Sig2Q = 0.7;
    _ratioThreshold_Q2Sig = 0.7;
    _maximumNNL2Dist = INFINITY;
    _maximumNNHammingDist = numeric_limits<unsigned int>::max();

    //////////////////////////////////////////////////////////////////  Attributes for Signature ingestion
    BRISK_sigLen = 0;
    ORB_sigLen = 0;
    SIFT_sigLen = 0;
    SURF_sigLen = 0;
                                                                    //  Initialize a Hamming-matcher for BRISK
    BRISK_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                           cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize a Hamming-matcher for ORB
    ORB_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                         cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SIFT
    SIFT_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                          cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SURF
    SURF_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                          cv::makePtr<cv::flann::SearchParams>());

    //////////////////////////////////////////////////////////////////  Attributes for Signature ingestion
    BRISK_QLen = 0;
    ORB_QLen = 0;
    SIFT_QLen = 0;
    SURF_QLen = 0;

                                                                    //  Initialize a Hamming-matcher for BRISK
    BRISK_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                         cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize a Hamming-matcher for ORB
    ORB_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                       cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SIFT
    SIFT_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                        cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SURF
    SURF_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                        cv::makePtr<cv::flann::SearchParams>());

    //////////////////////////////////////////////////////////////////  Initially, no matches
    numMatches = 0;
  }

Matcher::Matcher(PnPConfig config)
  {
    numSignatures = 0;                                              //  Initially, no Signatures

    _numKDtrees = config.numKDtrees();                              //  Take values from PnPConfig object
    _numTables = config.numTables();
    _topK = config.topK();
    _ratioThreshold_Sig2Q = config.ratioThresholdSig2Q();
    _ratioThreshold_Q2Sig = config.ratioThresholdQ2Sig();

    _maximumNNL2Dist = config.maximumNNL2Dist();
    _maximumNNHammingDist = config.maximumNNHammingDist();

    //////////////////////////////////////////////////////////////////  Attributes for Signature ingestion
    BRISK_sigLen = 0;
    ORB_sigLen = 0;
    SIFT_sigLen = 0;
    SURF_sigLen = 0;
                                                                    //  Initialize a Hamming-matcher for BRISK
    BRISK_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                           cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize a Hamming-matcher for ORB
    ORB_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                         cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SIFT
    SIFT_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                          cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SURF
    SURF_SigIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                          cv::makePtr<cv::flann::SearchParams>());

    //////////////////////////////////////////////////////////////////  Attributes for Signature ingestion
    BRISK_QLen = 0;
    ORB_QLen = 0;
    SIFT_QLen = 0;
    SURF_QLen = 0;

                                                                    //  Initialize a Hamming-matcher for BRISK
    BRISK_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                         cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize a Hamming-matcher for ORB
    ORB_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(_numTables, 20, 2),
                                       cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SIFT
    SIFT_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                        cv::makePtr<cv::flann::SearchParams>());

                                                                    //  Initialize an L2-matcher for SURF
    SURF_QIndex = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(_numKDtrees),
                                        cv::makePtr<cv::flann::SearchParams>());

    //////////////////////////////////////////////////////////////////  Initially, no matches
    numMatches = 0;
  }

/**************************************************************************************************
 Destructor  */

Matcher::~Matcher()
  {
  }

/**************************************************************************************************
 "Training"  */

/* This trains the FLANN-Indices for signatures on the given Signatures. */
unsigned int Matcher::train(Signature** signatures, unsigned int signaturesLen)
  {
    unsigned int features = 0;                                      //  Return total number of features

    unsigned char* ucharBuffer;                                     //  Accumulator for all uchar-descriptors in all Signatures
    float* floatBuffer;                                             //  Accumulator for all float-descriptors in all Signatures

    void* signatureBuffer;                                          //  Accumulator for descriptors in a single Signature (uchar)

    cv::Mat DBMat_BRISK;                                            //  Massive tables of all Signatures' descriptor vectors: N x D
    cv::Mat DBMat_ORB;                                              //  where D is the length of the descriptor (say, 64 for BRISK)
    cv::Mat DBMat_SIFT;                                             //  and N is the number of features of this type over all Signatures.
    cv::Mat DBMat_SURF;

    unsigned int n, m, len;
    unsigned int i;
    unsigned char j;

    #ifdef __MATCHER_DEBUG
    cout << "Matcher::train(" << +signaturesLen << " signatures):" << endl;
    #endif

    numSignatures = signaturesLen;                                  //  Save number of Signatures

                                                                    //  Outer pass: count all Signatures' features in type 'j'
    for(j = 0; j < _DESCRIPTOR_TOTAL; j++)                          //  For each descriptor type...
      {
        n = 0;                                                      //  Count up the number of Descriptors of that type
        for(i = 0; i < signaturesLen; i++)                          //  across all Signatures
          n += signatures[i]->count(j);

        if(n > 0)                                                   //  If any Signatures have any Descriptors of type 'j'
          {
            features += n;                                          //  Add to total feature count

            switch(j)
              {
                case _BRISK:  BRISK_sigLen = n;                     //  Save total number of BRISK features
                                                                    //  Allocate Signature lookup
                              if((BRISK_signatureLookup = (unsigned int*)malloc(BRISK_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for BRISK features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((BRISK_signatureFeature = (unsigned int*)malloc(BRISK_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for BRISK." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate uchar buffer for ALL Signatures
                              if((ucharBuffer = (unsigned char*)malloc(BRISK_sigLen * _BRISK_DESCLEN * sizeof(char))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate uchar buffer for BRISK training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into both buffer and map
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signature's BRISK features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_BRISK, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s BRISK into BRISK accumulator buffer
                                    ucharBuffer[n * _BRISK_DESCLEN + m] = ((unsigned char*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_BRISK); m++)
                                    {
                                      BRISK_signatureLookup[n + m] = signatures[i]->index();
                                      BRISK_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_BRISK);//  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used buffer)
                                    free(signatureBuffer);
                                }
                                                                    //  Build BRISK matrix from buffer
                              DBMat_BRISK = cv::Mat(n, _BRISK_DESCLEN, CV_8U, ucharBuffer);
                              BRISK_SigIndex.add(DBMat_BRISK);      //  Build BRISK index from matrix
                              BRISK_SigIndex.train();               //  Tell the index, "That's all."
                              break;

                case _ORB:    ORB_sigLen = n;                       //  Save total number of ORB features
                                                                    //  Allocate Signature lookup
                              if((ORB_signatureLookup = (unsigned int*)malloc(ORB_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for ORB features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((ORB_signatureFeature = (unsigned int*)malloc(ORB_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for ORB." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate uchar buffer for ALL Signatures
                              if((ucharBuffer = (unsigned char*)malloc(ORB_sigLen * _ORB_DESCLEN * sizeof(char))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate uchar buffer for ORB training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into both buffer and map
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signatures' ORB features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_ORB, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s ORB into ORB accumulator buffer
                                    ucharBuffer[n * _ORB_DESCLEN + m] = ((unsigned char*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_ORB); m++)
                                    {
                                      ORB_signatureLookup[n + m] = signatures[i]->index();
                                      ORB_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_ORB);  //  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used buffer)
                                    free(signatureBuffer);
                                }
                                                                    //  Build ORB matrix from buffer
                              DBMat_ORB = cv::Mat(n, _ORB_DESCLEN, CV_8U, ucharBuffer);
                              ORB_SigIndex.add(DBMat_ORB);          //  Build ORB index from matrix
                              ORB_SigIndex.train();                 //  Tell the index, "That's all."
                              break;

                case _SIFT:   SIFT_sigLen = n;                      //  Save total number of SIFT features
                                                                    //  Allocate Signature lookup
                              if((SIFT_signatureLookup = (unsigned int*)malloc(SIFT_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for SIFT features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((SIFT_signatureFeature = (unsigned int*)malloc(SIFT_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for SIFT." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate float buffer for ALL Signatures
                              if((floatBuffer = (float*)malloc(SIFT_sigLen * _SIFT_DESCLEN * sizeof(float))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate float buffer for SIFT training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into map (and buffer)
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signatures' SIFT features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_SIFT, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s SIFT into SIFT accumulator buffer
                                    floatBuffer[n * _SIFT_DESCLEN + m] = ((float*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_SIFT); m++)
                                    {
                                      SIFT_signatureLookup[n + m] = signatures[i]->index();
                                      SIFT_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_SIFT); //  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used buffer)
                                    free(signatureBuffer);
                                }
                                                                    //  Build SIFT matrix from buffer
                              DBMat_SIFT = cv::Mat(n, _SIFT_DESCLEN, CV_32F, floatBuffer);
                              SIFT_SigIndex.add(DBMat_SIFT);        //  Build SIFT index from matrix
                              SIFT_SigIndex.train();                //  Tell the index, "That's all."
                              break;

                case _SURF:   SURF_sigLen = n;                      //  Save total number of SURF features
                                                                    //  Allocate Signature lookup
                              if((SURF_signatureLookup = (unsigned int*)malloc(SURF_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for SURF features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((SURF_signatureFeature = (unsigned int*)malloc(SURF_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for SURF." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate float buffer for ALL Signatures
                              if((floatBuffer = (float*)malloc(SURF_sigLen * _SURF_DESCLEN * sizeof(float))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate float buffer for SURF training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into both buffer and map
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signatures' SURF features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_SURF, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s SURF into SURF accumulator buffer
                                    floatBuffer[n * _SURF_DESCLEN + m] = ((float*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_SURF); m++)
                                    {
                                      SURF_signatureLookup[n + m] = signatures[i]->index();
                                      SURF_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_SURF); //  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used it)
                                    free(signatureBuffer);
                                }
                                                                    //  Build SURF matrix from buffer
                              DBMat_SURF = cv::Mat(n, _SURF_DESCLEN, CV_32F, floatBuffer);
                              SURF_SigIndex.add(DBMat_SURF);        //  Build SURF index from matrix
                              SURF_SigIndex.train();                //  Tell the index, "That's all."
                              break;
              }
          }
      }

    #ifdef __MATCHER_DEBUG
    cout << "  BRISK: " << +BRISK_sigLen << endl;
    cout << "  ORB: "   << +ORB_sigLen   << endl;
    cout << "  SIFT: "  << +SIFT_sigLen  << endl;
    cout << "  SURF: "  << +SURF_sigLen  << endl;
    #endif

    return features;
  }

/* This trains the FLANN-Indices for the query image using the given Extractor */
unsigned int Matcher::train(Extractor* extractor)
  {
    unsigned int features = 0;                                      //  Return total number of features

    cv::Mat DBMat_BRISK;                                            //  Massive tables of all Signatures' descriptor vectors: N x D
    cv::Mat DBMat_ORB;                                              //  where D is the length of the descriptor (say, 64 for BRISK)
    cv::Mat DBMat_SIFT;                                             //  and N is the number of features of this type over all Signatures.
    cv::Mat DBMat_SURF;

    unsigned int n;
    unsigned char j;

    #ifdef __MATCHER_DEBUG
    cout << "Matcher::train(extractor):" << endl;
    #endif
                                                                    //  Outer pass: count all Signatures' features in type 'j'
    for(j = 0; j < _DESCRIPTOR_TOTAL; j++)                          //  For each descriptor type...
      {
        switch(j)
          {
            case _BRISK:  if(extractor->BRISK())
                            {
                              n = extractor->features(_BRISK);
                              features += n;
                              if(n > 0)
                                {
                                  BRISK_QLen = n;                   //  Save total number of BRISK features
                                                                    //  Build BRISK matrix from Extractor
                                  extractor->descMat(_BRISK, &DBMat_BRISK);
                                  BRISK_QIndex.add(DBMat_BRISK);    //  Build BRISK index from matrix
                                  BRISK_QIndex.train();             //  Tell the index, "That's all."
                                }
                            }
                          break;
            case _ORB:    if(extractor->ORB())
                            {
                              n = extractor->features(_ORB);
                              features += n;
                              if(n > 0)
                                {
                                  ORB_QLen = n;                     //  Save total number of ORB features
                                                                    //  Build ORB matrix from Extractor
                                  extractor->descMat(_ORB, &DBMat_ORB);
                                  ORB_QIndex.add(DBMat_ORB);        //  Build ORB index from matrix
                                  ORB_QIndex.train();               //  Tell the index, "That's all."
                                }
                            }
                          break;
            case _SIFT:   if(extractor->SIFT())
                            {
                              n = extractor->features(_SIFT);
                              features += n;
                              if(n > 0)
                                {
                                  SIFT_QLen = n;                    //  Save total number of SIFT features
                                                                    //  Build SIFT matrix from Extractor
                                  extractor->descMat(_SIFT, &DBMat_SIFT);
                                  SIFT_QIndex.add(DBMat_SIFT);      //  Build SIFT index from matrix
                                  SIFT_QIndex.train();              //  Tell the index, "That's all."
                                }
                            }
                          break;
            case _SURF:   if(extractor->SURF())
                            {
                              n = extractor->features(_SURF);
                              features += n;
                              if(n > 0)
                                {
                                  SURF_QLen = n;                    //  Save total number of SURF features
                                                                    //  Build SURF matrix from Extractor
                                  extractor->descMat(_SURF, &DBMat_SURF);
                                  SURF_QIndex.add(DBMat_SURF);      //  Build SURF index from matrix
                                  SURF_QIndex.train();              //  Tell the index, "That's all."
                                }
                            }
                          break;
          }
      }

    #ifdef __MATCHER_DEBUG
    cout << "  BRISK: " << +BRISK_QLen << endl;
    cout << "  ORB: "   << +ORB_QLen   << endl;
    cout << "  SIFT: "  << +SIFT_QLen  << endl;
    cout << "  SURF: "  << +SURF_QLen  << endl;
    #endif

    return features;
  }

/**************************************************************************************************
 Matching  */

/* Match the FLANN-Index trained on Signatures to the Query image given by Extractor.
   Assumes that both Matcher.train(Signature**, uint) and Extractor.extract() have already been called.
   Extractor therefore has Descriptors in its internal array. */
unsigned int Matcher::match(Extractor* extractor)
  {
    cv::Mat query;                                                  //  N x D for each type of Descriptor the Extractor used
    std::vector< std::vector<DMatch> > knnMatches;
    unsigned int i;

    #ifdef __MATCHER_DEBUG
    cout << "Matcher::match()" << endl;
    #endif

    numMatches = 0;

    //////////////////////////////////////////////////////////////////  First pass: count the number of matches we will have

    if(extractor->features(_BRISK) >= 2 && BRISK_sigLen >= 2)       //  The Extractor contains BRISK *AND* this Matcher has trained on BRISK
      {
        extractor->descMat(_BRISK, &query);
        BRISK_SigIndex.knnMatch(query, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              numMatches++;
          }
        knnMatches.clear();
      }
    if(extractor->features(_ORB) >= 2 && ORB_sigLen >= 2)           //  The Extractor contains ORB *AND* this Matcher has trained on ORB
      {
        extractor->descMat(_ORB, &query);
        ORB_SigIndex.knnMatch(query, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              numMatches++;
          }
        knnMatches.clear();
      }
    if(extractor->features(_SIFT) >= 2 && SIFT_sigLen >= 2)         //  The Extractor contains SIFT *AND* this Matcher has trained on SIFT
      {
        extractor->descMat(_SIFT, &query);
        SIFT_SigIndex.knnMatch(query, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              numMatches++;
          }
        knnMatches.clear();
      }
    if(extractor->features(_SURF) >= 2 && SURF_sigLen >= 2)         //  The Extractor contains SURF *AND* this Matcher has trained on SURF
      {
        extractor->descMat(_SURF, &query);
        SURF_SigIndex.knnMatch(query, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              numMatches++;
          }
        knnMatches.clear();
      }

    //////////////////////////////////////////////////////////////////  Second pass: allocate and write matches
    if(numMatches > 0)
      {
        if((matches = (Match*)malloc(numMatches * sizeof(Match))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate array of Match structs." << endl;
            #endif
            return 0;
          }

        numMatches = 0;                                             //  Reset

        if(extractor->features(_BRISK) >= 2 && BRISK_sigLen >= 2)
          {
            extractor->descMat(_BRISK, &query);
            BRISK_SigIndex.knnMatch(query, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, BRISK-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _BRISK;
                    matches[numMatches].signature    = BRISK_signatureLookup[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].sigFeature   = BRISK_signatureFeature[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].queryIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
        if(extractor->features(_ORB) >= 2 && ORB_sigLen >= 2)
          {
            extractor->descMat(_ORB, &query);
            ORB_SigIndex.knnMatch(query, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, ORB-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _ORB;
                    matches[numMatches].signature    = ORB_signatureLookup[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].sigFeature   = ORB_signatureFeature[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].queryIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
        if(extractor->features(_SIFT) >= 2 && SIFT_sigLen >= 2)
          {
            extractor->descMat(_SIFT, &query);
            SIFT_SigIndex.knnMatch(query, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, SIFT-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _SIFT;
                    matches[numMatches].signature    = SIFT_signatureLookup[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].sigFeature   = SIFT_signatureFeature[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].queryIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
        if(extractor->features(_SURF) >= 2 && SURF_sigLen >= 2)
          {
            extractor->descMat(_SURF, &query);
            SURF_SigIndex.knnMatch(query, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, SURF-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _SURF;
                    matches[numMatches].signature    = SURF_signatureLookup[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].sigFeature   = SURF_signatureFeature[ knnMatches[i][0].trainIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].queryIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
      }

    return numMatches;
  }

/* Match the FLANN-Index trained on the Query image to the given Signatures.
   Assumes that Extractor.extract() has been called and that Matcher.train(Extractor) has been called. */
unsigned int Matcher::match(Signature** signatures, unsigned int signaturesLen)
  {
    unsigned char* ucharBuffer;                                     //  Accumulator for all uchar-descriptors in all Signatures
    float* floatBuffer;                                             //  Accumulator for all float-descriptors in all Signatures

    void* signatureBuffer;                                          //  Accumulator for descriptors in a single Signature (uchar)

    cv::Mat SigsMat_BRISK;                                          //  Massive tables of all Signatures' descriptor vectors: N x D
    cv::Mat SigsMat_ORB;                                            //  where D is the length of the descriptor (say, 64 for BRISK)
    cv::Mat SigsMat_SIFT;                                           //  and N is the number of features of this type over all Signatures.
    cv::Mat SigsMat_SURF;
    unsigned int num_BRISK = 0;
    unsigned int num_ORB = 0;
    unsigned int num_SIFT = 0;
    unsigned int num_SURF = 0;

    std::vector< std::vector<DMatch> > knnMatches;

    unsigned int n, m, len;
    unsigned int i;
    unsigned char j;

    #ifdef __MATCHER_DEBUG
    cout << "Matcher::match(" << +signaturesLen << " signatures):" << endl;
    #endif

    numMatches = 0;
    numSignatures = signaturesLen;                                  //  Save number of Signatures

                                                                    //  Outer pass: count all Signatures' features in type 'j'
    for(j = 0; j < _DESCRIPTOR_TOTAL; j++)                          //  For each descriptor type...
      {
        n = 0;                                                      //  Count up the number of Descriptors of that type
        for(i = 0; i < signaturesLen; i++)                          //  across all Signatures
          n += signatures[i]->count(j);

        if(n > 0)                                                   //  If any Signatures have any Descriptors of type 'j'
          {
            switch(j)
              {
                case _BRISK:  BRISK_sigLen = n;                     //  Save total number of BRISK features
                                                                    //  Allocate Signature lookup
                              if((BRISK_signatureLookup = (unsigned int*)malloc(BRISK_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for BRISK features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((BRISK_signatureFeature = (unsigned int*)malloc(BRISK_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for BRISK." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate uchar buffer for ALL Signatures
                              if((ucharBuffer = (unsigned char*)malloc(BRISK_sigLen * _BRISK_DESCLEN * sizeof(char))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate uchar buffer for BRISK training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into both buffer and map
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signature's BRISK features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_BRISK, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s BRISK into BRISK accumulator buffer
                                    ucharBuffer[n * _BRISK_DESCLEN + m] = ((unsigned char*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_BRISK); m++)
                                    {
                                      BRISK_signatureLookup[n + m] = signatures[i]->index();
                                      BRISK_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_BRISK);//  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used buffer)
                                    free(signatureBuffer);
                                }
                                                                    //  Build BRISK matrix from buffer
                              num_BRISK = n;
                              SigsMat_BRISK = cv::Mat(num_BRISK, _BRISK_DESCLEN, CV_8U, ucharBuffer);
                              break;

                case _ORB:    ORB_sigLen = n;                       //  Save total number of ORB features
                                                                    //  Allocate Signature lookup
                              if((ORB_signatureLookup = (unsigned int*)malloc(ORB_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for ORB features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((ORB_signatureFeature = (unsigned int*)malloc(ORB_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for ORB." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate uchar buffer for ALL Signatures
                              if((ucharBuffer = (unsigned char*)malloc(ORB_sigLen * _ORB_DESCLEN * sizeof(char))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate uchar buffer for ORB training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into both buffer and map
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signatures' ORB features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_ORB, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s ORB into ORB accumulator buffer
                                    ucharBuffer[n * _ORB_DESCLEN + m] = ((unsigned char*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_ORB); m++)
                                    {
                                      ORB_signatureLookup[n + m] = signatures[i]->index();
                                      ORB_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_ORB);  //  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used buffer)
                                    free(signatureBuffer);
                                }
                                                                    //  Build ORB matrix from buffer
                              num_ORB = n;
                              SigsMat_ORB = cv::Mat(num_ORB, _ORB_DESCLEN, CV_8U, ucharBuffer);
                              break;

                case _SIFT:   SIFT_sigLen = n;                      //  Save total number of SIFT features
                                                                    //  Allocate Signature lookup
                              if((SIFT_signatureLookup = (unsigned int*)malloc(SIFT_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for SIFT features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((SIFT_signatureFeature = (unsigned int*)malloc(SIFT_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for SIFT." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate float buffer for ALL Signatures
                              if((floatBuffer = (float*)malloc(SIFT_sigLen * _SIFT_DESCLEN * sizeof(float))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate float buffer for SIFT training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into map (and buffer)
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signatures' SIFT features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_SIFT, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s SIFT into SIFT accumulator buffer
                                    floatBuffer[n * _SIFT_DESCLEN + m] = ((float*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_SIFT); m++)
                                    {
                                      SIFT_signatureLookup[n + m] = signatures[i]->index();
                                      SIFT_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_SIFT); //  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used buffer)
                                    free(signatureBuffer);
                                }
                                                                    //  Build SIFT matrix from buffer
                              num_SIFT = n;
                              SigsMat_SIFT = cv::Mat(num_SIFT, _SIFT_DESCLEN, CV_32F, floatBuffer);
                              break;

                case _SURF:   SURF_sigLen = n;                      //  Save total number of SURF features
                                                                    //  Allocate Signature lookup
                              if((SURF_signatureLookup = (unsigned int*)malloc(SURF_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate Signature-lookup array for SURF features." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate individual Signature feature indices
                              if((SURF_signatureFeature = (unsigned int*)malloc(SURF_sigLen * sizeof(int))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate feature-index array for SURF." << endl;
                                  #endif
                                  return 0;
                                }
                                                                    //  Allocate float buffer for ALL Signatures
                              if((floatBuffer = (float*)malloc(SURF_sigLen * _SURF_DESCLEN * sizeof(float))) == NULL)
                                {
                                  #ifdef __MATCHER_DEBUG
                                  cout << "ERROR: Unable to allocate float buffer for SURF training." << endl;
                                  #endif
                                  return 0;
                                }

                              n = 0;                                //  'n' now acts as an offset into both buffer and map
                              for(i = 0; i < signaturesLen; i++)    //  Collect each Signatures' SURF features
                                {
                                                                    //  'len' is N x D
                                  len = signatures[i]->toBuffer(_SURF, &signatureBuffer);
                                  for(m = 0; m < len; m++)          //  Copy Signature[i]'s SURF into SURF accumulator buffer
                                    floatBuffer[n * _SURF_DESCLEN + m] = ((float*)signatureBuffer)[m];
                                  for(m = 0; m < signatures[i]->count(_SURF); m++)
                                    {
                                      SURF_signatureLookup[n + m] = signatures[i]->index();
                                      SURF_signatureFeature[n + m] = m;
                                    }
                                  n += signatures[i]->count(_SURF); //  Increase offset
                                  if(len > 0)                       //  Release buffer (if we used it)
                                    free(signatureBuffer);
                                }
                                                                    //  Build SURF matrix from buffer
                              num_SURF = n;
                              SigsMat_SURF = cv::Mat(num_SURF, _SURF_DESCLEN, CV_32F, floatBuffer);
                              break;
              }
          }
      }

    //////////////////////////////////////////////////////////////////  First pass: count the number of matches we will have

    if(BRISK_QLen >= 2 && num_BRISK >= 2)                           //  Matcher has trained on BRISK AND Signatures contain BRISK
      {
        BRISK_QIndex.knnMatch(SigsMat_BRISK, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              numMatches++;
          }
        knnMatches.clear();
      }
    if(ORB_QLen >= 2 && num_ORB >= 2)                               //  Matcher has trained on ORB AND Signatures contain ORB
      {
        ORB_QIndex.knnMatch(SigsMat_ORB, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              numMatches++;
          }
        knnMatches.clear();
      }
    if(SIFT_QLen >= 2 && num_SIFT >= 2)                             //  Matcher has trained on SIFT AND Signatures contain SIFT
      {
        SIFT_QIndex.knnMatch(SigsMat_SIFT, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              numMatches++;
          }
        knnMatches.clear();
      }
    if(SURF_QLen >= 2 && num_SURF >= 2)                             //  Matcher has trained on SURF AND Signatures contain SURF
      {
        SURF_QIndex.knnMatch(SigsMat_SURF, knnMatches, 2);
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              numMatches++;
          }
        knnMatches.clear();
      }

    //////////////////////////////////////////////////////////////////  Second pass: allocate and write matches

    if(numMatches > 0)
      {
        if((matches = (Match*)malloc(numMatches * sizeof(Match))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate array of Match structs." << endl;
            #endif
            return 0;
          }

        numMatches = 0;                                             //  Reset

        if(BRISK_QLen >= 2 && num_BRISK >= 2)
          {
            BRISK_QIndex.knnMatch(SigsMat_BRISK, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, BRISK-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _BRISK;
                    matches[numMatches].signature    = BRISK_signatureLookup[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].sigFeature   = BRISK_signatureFeature[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].trainIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
        if(ORB_QLen >= 2 && num_ORB >= 2)
          {
            ORB_QIndex.knnMatch(SigsMat_ORB, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, ORB-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _ORB;
                    matches[numMatches].signature    = ORB_signatureLookup[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].sigFeature   = ORB_signatureFeature[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].trainIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
        if(SIFT_QLen >= 2 && num_SIFT >= 2)
          {
            SIFT_QIndex.knnMatch(SigsMat_SIFT, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, SIFT-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _SIFT;
                    matches[numMatches].signature    = SIFT_signatureLookup[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].sigFeature   = SIFT_signatureFeature[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].trainIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
        if(SURF_QLen >= 2 && num_SURF >= 2)
          {
            SURF_QIndex.knnMatch(SigsMat_SURF, knnMatches, 2);
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    #ifdef __MATCHER_DEBUG
                    cout << "Distance between nearest neighbors, SURF-match[" << +i << "] = " << knnMatches[i][0].distance << endl;
                    #endif

                    matches[numMatches].type         = _SURF;
                    matches[numMatches].signature    = SURF_signatureLookup[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].sigFeature   = SURF_signatureFeature[ knnMatches[i][0].queryIdx ];
                    matches[numMatches].queryFeature = knnMatches[i][0].trainIdx;

                    numMatches++;
                  }
              }
            knnMatches.clear();
          }
      }

    return numMatches;
  }

/* Match the FLANN-Index trained on Signatures to the Query image features, and
   match the FLANN-Index trained on the Query image to the Signature features.

   Count as matches those which are mutually nearest neighbors. */
unsigned int Matcher::match()
  {
    std::vector< std::vector<DMatch> > knnMatches;
    unsigned int i, j;

    std::vector<cv::Mat> SigsMat_BRISK;                             //  Massive tables of all Signatures' descriptor vectors: N x D
    std::vector<cv::Mat> SigsMat_ORB;                               //  where D is the length of the descriptor (say, 64 for BRISK)
    std::vector<cv::Mat> SigsMat_SIFT;                              //  and N is the number of features of this type over all Signatures.
    std::vector<cv::Mat> SigsMat_SURF;

    std::vector<cv::Mat> QMat_BRISK;                                //  Massive tables of all Query's descriptor vectors: N x D
    std::vector<cv::Mat> QMat_ORB;                                  //  where D is the length of the descriptor (say, 64 for BRISK)
    std::vector<cv::Mat> QMat_SIFT;                                 //  and N is the number of features of this type in the Query image.
    std::vector<cv::Mat> QMat_SURF;

    Match* Sig2Q_matches;                                           //  Array of half the tentative matches
    unsigned int Sig2Q_matchesLen = 0;

    Match* Q2Sig_matches;                                           //  The array of the other half of tentative matches
    unsigned int Q2Sig_matchesLen = 0;

    #ifdef __MATCHER_DEBUG
    cout << "Matcher::match():" << endl;
    #endif

    numMatches = 0;                                                 //  Remember, this is the TOTAL number of FINAL matches

    //////////////////////////////////////////////////////////////////  First pass: a count-up
    if(BRISK_sigLen >= 2 && BRISK_QLen >= 2)
      {
        SigsMat_BRISK = BRISK_SigIndex.getTrainDescriptors();       //  Fetch the (BRISK_sigLen x _BRISK_LEN) matrix of descriptor vectors
        QMat_BRISK = BRISK_QIndex.getTrainDescriptors();            //  Fetch the (BRISK_QLen x _BRISK_LEN) matrix of descriptor vectors

        BRISK_SigIndex.knnMatch(QMat_BRISK[0], knnMatches, 2);      //  Match Signatures against Query
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              Sig2Q_matchesLen++;
          }
        knnMatches.clear();

        BRISK_QIndex.knnMatch(SigsMat_BRISK[0], knnMatches, 2);     //  Match Query against Signatures
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              Q2Sig_matchesLen++;
          }
        knnMatches.clear();
      }
    if(ORB_sigLen >= 2 && ORB_QLen >= 2)
      {
        SigsMat_ORB = ORB_SigIndex.getTrainDescriptors();           //  Fetch the (ORB_sigLen x _ORB_LEN) matrix of descriptor vectors
        QMat_ORB = ORB_QIndex.getTrainDescriptors();                //  Fetch the (ORB_QLen x _ORB_LEN) matrix of descriptor vectors

        ORB_SigIndex.knnMatch(QMat_ORB[0], knnMatches, 2);          //  Match Signatures against Query
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              Sig2Q_matchesLen++;
          }
        knnMatches.clear();

        ORB_QIndex.knnMatch(SigsMat_ORB[0], knnMatches, 2);         //  Match Query against Signatures
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNHammingDist)
              Q2Sig_matchesLen++;
          }
        knnMatches.clear();
      }
    if(SIFT_sigLen >= 2 && SIFT_QLen >= 2)
      {
        SigsMat_SIFT = SIFT_SigIndex.getTrainDescriptors();         //  Fetch the (SIFT_sigLen x _SIFT_LEN) matrix of descriptor vectors
        QMat_SIFT = SIFT_QIndex.getTrainDescriptors();              //  Fetch the (SIFT_QLen x _SIFT_LEN) matrix of descriptor vectors

        SIFT_SigIndex.knnMatch(QMat_SIFT[0], knnMatches, 2);        //  Match Signatures against Query
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              Sig2Q_matchesLen++;
          }
        knnMatches.clear();

        SIFT_QIndex.knnMatch(SigsMat_SIFT[0], knnMatches, 2);       //  Match Query against Signatures
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              Q2Sig_matchesLen++;
          }
        knnMatches.clear();
      }
    if(SURF_sigLen >= 2 && SURF_QLen >= 2)
      {
        SigsMat_SURF = SURF_SigIndex.getTrainDescriptors();         //  Fetch the (SURF_sigLen x _SURF_LEN) matrix of descriptor vectors
        QMat_SURF = SURF_QIndex.getTrainDescriptors();              //  Fetch the (SURF_QLen x _SURF_LEN) matrix of descriptor vectors

        SURF_SigIndex.knnMatch(QMat_SURF[0], knnMatches, 2);        //  Match Signatures against Query
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              Sig2Q_matchesLen++;
          }
        knnMatches.clear();

        SURF_QIndex.knnMatch(SigsMat_SURF[0], knnMatches, 2);       //  Match Query against Signatures
        for(i = 0; i < (unsigned int)knnMatches.size(); i++)
          {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
            if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                         && knnMatches[i][0].distance < _maximumNNL2Dist)
              Q2Sig_matchesLen++;
          }
        knnMatches.clear();
      }

    //////////////////////////////////////////////////////////////////  Second pass: allocate and write
    if(Sig2Q_matchesLen > 0 && Q2Sig_matchesLen > 0)
      {
        if((Sig2Q_matches = (Match*)malloc(Sig2Q_matchesLen * sizeof(Match))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate Signature-to-Query matches." << endl;
            #endif
            return 0;
          }
        if((Q2Sig_matches = (Match*)malloc(Q2Sig_matchesLen * sizeof(Match))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate Query-to-Signature matches." << endl;
            #endif
            free(Sig2Q_matches);
            return 0;
          }

        Sig2Q_matchesLen = 0;                                       //  Reset and use as indices into Match arrays
        Q2Sig_matchesLen = 0;

        if(BRISK_sigLen >= 2 && BRISK_QLen >= 2)                    //  Fill in BRISK matches, Sig-->Q + Q-->Sig
          {
            BRISK_SigIndex.knnMatch(QMat_BRISK[0], knnMatches, 2);  //  Match Signatures against Query
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    Sig2Q_matches[Sig2Q_matchesLen].type         = _BRISK;
                    Sig2Q_matches[Sig2Q_matchesLen].signature    = BRISK_signatureLookup[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].sigFeature   = BRISK_signatureFeature[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].queryFeature = knnMatches[i][0].queryIdx;
                    Sig2Q_matchesLen++;
                  }
              }
            knnMatches.clear();

            BRISK_QIndex.knnMatch(SigsMat_BRISK[0], knnMatches, 2); //  Match Query against Signatures
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    Q2Sig_matches[Q2Sig_matchesLen].type         = _BRISK;
                    Q2Sig_matches[Q2Sig_matchesLen].signature    = BRISK_signatureLookup[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].sigFeature   = BRISK_signatureFeature[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].queryFeature = knnMatches[i][0].trainIdx;
                    Q2Sig_matchesLen++;
                  }
              }
            knnMatches.clear();
          }
        if(ORB_sigLen >= 2 && ORB_QLen >= 2)                        //  Fill in ORB matches, Sig-->Q + Q-->Sig
          {
            ORB_SigIndex.knnMatch(QMat_ORB[0], knnMatches, 2);      //  Match Signatures against Query
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    Sig2Q_matches[Sig2Q_matchesLen].type         = _ORB;
                    Sig2Q_matches[Sig2Q_matchesLen].signature    = ORB_signatureLookup[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].sigFeature   = ORB_signatureFeature[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].queryFeature = knnMatches[i][0].queryIdx;
                    Sig2Q_matchesLen++;
                  }
              }
            knnMatches.clear();

            ORB_QIndex.knnMatch(SigsMat_ORB[0], knnMatches, 2);     //  Match Query against Signatures
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNHammingDist)
                  {
                    Q2Sig_matches[Q2Sig_matchesLen].type         = _ORB;
                    Q2Sig_matches[Q2Sig_matchesLen].signature    = ORB_signatureLookup[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].sigFeature   = ORB_signatureFeature[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].queryFeature = knnMatches[i][0].trainIdx;
                    Q2Sig_matchesLen++;
                  }
              }
            knnMatches.clear();
          }
        if(SIFT_sigLen >= 2 && SIFT_QLen >= 2)                      //  Fill in SIFT matches, Sig-->Q + Q-->Sig
          {
            SIFT_SigIndex.knnMatch(QMat_SIFT[0], knnMatches, 2);    //  Match Signatures against Query
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    Sig2Q_matches[Sig2Q_matchesLen].type         = _SIFT;
                    Sig2Q_matches[Sig2Q_matchesLen].signature    = SIFT_signatureLookup[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].sigFeature   = SIFT_signatureFeature[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].queryFeature = knnMatches[i][0].queryIdx;
                    Sig2Q_matchesLen++;
                  }
              }
            knnMatches.clear();

            SIFT_QIndex.knnMatch(SigsMat_SIFT[0], knnMatches, 2);   //  Match Query against Signatures
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    Q2Sig_matches[Q2Sig_matchesLen].type         = _SIFT;
                    Q2Sig_matches[Q2Sig_matchesLen].signature    = SIFT_signatureLookup[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].sigFeature   = SIFT_signatureFeature[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].queryFeature = knnMatches[i][0].trainIdx;
                    Q2Sig_matchesLen++;
                  }
              }
            knnMatches.clear();
          }
        if(SURF_sigLen >= 2 && SURF_QLen >= 2)                      //  Fill in SURF matches, Sig-->Q + Q-->Sig
          {
            SURF_SigIndex.knnMatch(QMat_SURF[0], knnMatches, 2);    //  Match Signatures against Query
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Sig2Q * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    Sig2Q_matches[Sig2Q_matchesLen].type         = _SURF;
                    Sig2Q_matches[Sig2Q_matchesLen].signature    = SURF_signatureLookup[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].sigFeature   = SURF_signatureFeature[ knnMatches[i][0].trainIdx ];
                    Sig2Q_matches[Sig2Q_matchesLen].queryFeature = knnMatches[i][0].queryIdx;
                    Sig2Q_matchesLen++;
                  }
              }
            knnMatches.clear();

            SURF_QIndex.knnMatch(SigsMat_SURF[0], knnMatches, 2);   //  Match Query against Signatures
            for(i = 0; i < (unsigned int)knnMatches.size(); i++)
              {
                                                                    //  Hamming-distance measurements will not always have
                                                                    //  two nearest neighbors
                if(knnMatches[i].size() >= 2 && knnMatches[i][0].distance < _ratioThreshold_Q2Sig * knnMatches[i][1].distance
                                             && knnMatches[i][0].distance < _maximumNNL2Dist)
                  {
                    Q2Sig_matches[Q2Sig_matchesLen].type         = _SURF;
                    Q2Sig_matches[Q2Sig_matchesLen].signature    = SURF_signatureLookup[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].sigFeature   = SURF_signatureFeature[ knnMatches[i][0].queryIdx ];
                    Q2Sig_matches[Q2Sig_matchesLen].queryFeature = knnMatches[i][0].trainIdx;
                    Q2Sig_matchesLen++;
                  }
              }
            knnMatches.clear();
          }

        //////////////////////////////////////////////////////////////  Third pass: count up mutual matches
        for(i = 0; i < Sig2Q_matchesLen; i++)                       //  Count up all Sig-->Q + Q-->Sig
          {
            j = 0;
            while(j < Q2Sig_matchesLen && !(Sig2Q_matches[i].type == Q2Sig_matches[j].type &&
                                            Sig2Q_matches[i].signature == Q2Sig_matches[j].signature &&
                                            Sig2Q_matches[i].sigFeature == Q2Sig_matches[j].sigFeature &&
                                            Sig2Q_matches[i].queryFeature == Q2Sig_matches[j].queryFeature ))
              j++;
            if(j < Q2Sig_matchesLen)
              numMatches++;
          }

        //////////////////////////////////////////////////////////////  Third pass: allocate and write the REAL (mutual) matches
        if(numMatches > 0)
          {
            if((matches = (Match*)malloc(numMatches * sizeof(Match))) == NULL)
              {
                #ifdef __MATCHER_DEBUG
                cout << "ERROR: Unable to allocate array of mutual matches." << endl;
                #endif
                free(Sig2Q_matches);
                free(Q2Sig_matches);
                return 0;
              }

            numMatches = 0;                                         //  Reset and use as a counter

            for(i = 0; i < Sig2Q_matchesLen; i++)                   //  Count up all Sig-->Q + Q-->Sig
              {
                j = 0;
                while(j < Q2Sig_matchesLen && !(Sig2Q_matches[i].type == Q2Sig_matches[j].type &&
                                                Sig2Q_matches[i].signature == Q2Sig_matches[j].signature &&
                                                Sig2Q_matches[i].sigFeature == Q2Sig_matches[j].sigFeature &&
                                                Sig2Q_matches[i].queryFeature == Q2Sig_matches[j].queryFeature ))
                  j++;
                if(j < Q2Sig_matchesLen)
                  {
                    matches[numMatches].type         = Sig2Q_matches[i].type;
                    matches[numMatches].signature    = Sig2Q_matches[i].signature;
                    matches[numMatches].sigFeature   = Sig2Q_matches[i].sigFeature;
                    matches[numMatches].queryFeature = Sig2Q_matches[i].queryFeature;
                    numMatches++;
                  }
              }
          }

        free(Sig2Q_matches);
        free(Q2Sig_matches);
      }

    return numMatches;
  }

/**************************************************************************************************
 Voting and Correspondence  */

/* Assuming matches have already been made using match(),
   this function writes the Signature indices of the 'topK' candidate objects to the given 'buffer'.
   Write the vote counts to the 'ballots' buffer.
   The returned value is the length of 'buffer'. */
unsigned int Matcher::top(unsigned int** buffer, unsigned int** ballots) const
  {
    unsigned int* indices;
    unsigned int* votes;
    unsigned int len = 0;
    unsigned int i;

    #ifdef __MATCHER_DEBUG
    cout << "Matcher::top()" << endl;
    #endif

    if(numMatches > 0)                                              //  If there are any matches
      {
                                                                    //  Allocate as many bins as there are Signatures
        if((votes = (unsigned int*)malloc(numSignatures * sizeof(int))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate array for object-match voting." << endl;
            #endif
            return 0;
          }
                                                                    //  Allocate bin indices we can track
        if((indices = (unsigned int*)malloc(numSignatures * sizeof(int))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate index array for object-match voting." << endl;
            #endif
            return 0;
          }
        for(i = 0; i < numSignatures; i++)                          //  Initialize all ballot counts to 0; label all bins
          {
            indices[i] = i;
            votes[i] = 0;
          }
        for(i = 0; i < numMatches; i++)                             //  Count up all match/votes
          votes[ matches[i].signature ]++;
        quicksort(&votes, &indices, 0, numSignatures - 1);          //  Sort the arrays, descending, according to votes
        len = (_topK > numSignatures) ? numSignatures : _topK;
        if(((*buffer) = (unsigned int*)malloc(len * sizeof(int))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate array for top candidates." << endl;
            #endif
            return 0;
          }
        if(((*ballots) = (unsigned int*)malloc(len * sizeof(int))) == NULL)
          {
            #ifdef __MATCHER_DEBUG
            cout << "ERROR: Unable to allocate ballot array for top candidates." << endl;
            #endif
            return 0;
          }
        for(i = 0 ; i < len; i++)
          {
            (*buffer)[i] = indices[i];
            (*ballots)[i] = votes[i];
          }
        free(votes);
        free(indices);
      }

    return len;
  }

/* Iterate over the array 'matches'. Look up and push corresponding points to vectors of 2D and 3D points. */
unsigned int Matcher::correspondences(Signature* signature, Extractor* extractor,
                                      std::vector<cv::Point2f>* pt2, std::vector<cv::Point3f>* pt3) const
  {
    unsigned int len = 0;
    unsigned int i;

    (*pt2).clear();
    (*pt3).clear();

    for(i = 0; i < numMatches; i++)
      {
        if(matches[i].signature == signature->index())              //  A Match struct relates to the given Signature
          {
            (*pt2).push_back( Point2f(extractor->x( matches[i].queryFeature ),
                                      extractor->y( matches[i].queryFeature )) );

            (*pt3).push_back( Point3f(signature->x( matches[i].sigFeature, matches[i].type ),
                                      signature->y( matches[i].sigFeature, matches[i].type ),
                                      signature->z( matches[i].sigFeature, matches[i].type )) );

            #ifdef __MATCHER_DEBUG
            cout << "  (" << (*pt2).at(len).x << ", " << (*pt2).at(len).y << ") --> ("
                          << (*pt3).at(len).x << ", " << (*pt3).at(len).y << ", " << (*pt3).at(len).z << ")" << endl;
            #endif

            len++;
          }
      }

    return len;
  }

/* Arrange both 'votes' and 'indices' according to values in 'votes'. */
void Matcher::quicksort(unsigned int** votes, unsigned int** indices, unsigned int lo, unsigned int hi) const
  {
    unsigned int p;

    if(lo < hi)
      {
        p = partition(true, votes, indices, lo, hi);                //  Sort descending by votes

        if(p > 0)                                                   //  PREVENT ROLL-OVER TO INT_MAX
          quicksort(votes, indices, lo, p - 1);                     //  Left side: start quicksort
        if(p < INT_MAX)                                             //  PREVENT ROLL-OVER TO 0
          quicksort(votes, indices, p + 1, hi);                     //  Right side: start quicksort
      }

    return;
  }

unsigned int Matcher::partition(bool desc, unsigned int** votes, unsigned int** indices, unsigned int lo, unsigned int hi) const
  {
    double pivot;
    unsigned int i = lo;
    unsigned int j;
    unsigned int tmpInt;
    bool trigger;

    pivot = (*votes)[hi];

    for(j = lo; j < hi; j++)
      {
        if(desc)
          trigger = ((*votes)[j] > pivot);                          //  SORT DESCENDING
        else
          trigger = ((*votes)[j] < pivot);                          //  SORT ASCENDING

        if(trigger)
          {
            tmpInt = (*indices)[i];                                 //  tmp gets [i]
            (*indices)[i] = (*indices)[j];                          //  [i] gets [j]
            (*indices)[j] = tmpInt;                                 //  [j] gets tmp

            tmpInt = (*votes)[i];                                   //  tmp gets [i]
            (*votes)[i] = (*votes)[j];                              //  [i] gets [j]
            (*votes)[j] = tmpInt;                                   //  [j] gets tmp

            i++;
          }
      }

    tmpInt = (*indices)[i];                                         //  tmp gets [i]
    (*indices)[i] = (*indices)[hi];                                 //  [i] gets [hi]
    (*indices)[hi] = tmpInt;                                        //  [hi] gets tmp

    tmpInt = (*votes)[i];                                           //  tmp gets [i]
    (*votes)[i] = (*votes)[hi];                                     //  [i] gets [hi]
    (*votes)[hi] = tmpInt;                                          //  [hi] gets tmp

    return i;
  }

/**************************************************************************************************
 Getters  */

unsigned int Matcher::numKDtrees(void) const
  {
    return _numKDtrees;
  }

unsigned int Matcher::numTables(void) const
  {
    return _numTables;
  }

unsigned int Matcher::topK(void) const
  {
    return _topK;
  }

float Matcher::ratio_Sig2Q(void) const
  {
    return _ratioThreshold_Sig2Q;
  }

float Matcher::ratio_Q2Sig(void) const
  {
    return _ratioThreshold_Q2Sig;
  }

float Matcher::maxL2(void) const
  {
    return _maximumNNL2Dist;
  }

unsigned int Matcher::maxHamming(void) const
  {
    return _maximumNNHammingDist;
  }

/**************************************************************************************************
 Reset  */

void Matcher::clearSig(void)                                          //  Reset the Signature indices only
  {
    #ifdef __MATCHER_DEBUG
    cout << "Matcher::clearSig()" << endl;
    #endif

    BRISK_SigIndex.clear();
    ORB_SigIndex.clear();
    SIFT_SigIndex.clear();
    SURF_SigIndex.clear();

    if(BRISK_sigLen > 0)
      {
        free(BRISK_signatureLookup);
        free(BRISK_signatureFeature);
        BRISK_sigLen = 0;
      }
    if(ORB_sigLen > 0)
      {
        free(ORB_signatureLookup);
        free(ORB_signatureFeature);
        ORB_sigLen = 0;
      }
    if(SIFT_sigLen > 0)
      {
        free(SIFT_signatureLookup);
        free(SIFT_signatureFeature);
        SIFT_sigLen = 0;
      }
    if(SURF_sigLen > 0)
      {
        free(SURF_signatureLookup);
        free(SURF_signatureFeature);
        SURF_sigLen = 0;
      }
  }

void Matcher::clearQ(void)                                            //  Reset the Query indices only
  {
    #ifdef __MATCHER_DEBUG
    cout << "Matcher::clearQ()" << endl;
    #endif

    BRISK_QIndex.clear();
    ORB_QIndex.clear();
    SIFT_QIndex.clear();
    SURF_QIndex.clear();

    BRISK_QLen = 0;
    ORB_QLen = 0;
    SIFT_QLen = 0;
    SURF_QLen = 0;
  }

void Matcher::clear(void)
  {
    #ifdef __MATCHER_DEBUG
    cout << "Matcher::clear()" << endl;
    #endif

    clearSig();
    clearQ();

    return;
  }

/**************************************************************************************************
 Display  */

void Matcher::printSignatureFeatures(void) const
  {
    unsigned int i;
    unsigned char j;

    for(j = 0; j < _DESCRIPTOR_TOTAL; j++)
      {
        switch(j)
          {
            case _BRISK:  for(i = 0; i < BRISK_sigLen; i++)
                            cout << "BRISK " << +i << ": object " << +BRISK_signatureLookup[i] << endl;
                          break;
            case _ORB:    for(i = 0; i < ORB_sigLen; i++)
                            cout << "ORB " << +i << ": object " << +ORB_signatureLookup[i] << endl;
                          break;
            case _SIFT:   for(i = 0; i < SIFT_sigLen; i++)
                            cout << "SIFT " << +i << ": object " << +SIFT_signatureLookup[i] << endl;
                          break;
            case _SURF:   for(i = 0; i < SURF_sigLen; i++)
                            cout << "SURF " << +i << ": object " << +SURF_signatureLookup[i] << endl;
                          break;
          }
      }

    return;
  }

#endif