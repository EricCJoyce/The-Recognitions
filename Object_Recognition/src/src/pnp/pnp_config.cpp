#ifndef __PNP_CONFIG_CPP
#define __PNP_CONFIG_CPP

#include "pnp_config.h"

/**************************************************************************************************
 Constructors  */

/* PnPConfig constructor, no data given */
PnPConfig::PnPConfig()
  {
    #ifdef __PNP_CONFIG_DEBUG
    cout << "PnPConfig::PnPConfig()" << endl;
    #endif

    QFPlen = 0;
    SigFPlen = 0;
    KFNlen = 0;
    RtFNlen = 0;
    distortionFNlen = 0;
    SigFPIsDirectory = false;
    maskFPlen = 0;

    _iterations = 2000;
    _reprojectionError = 2.0;
    _confidence = 0.9999;
    _ransacMethod = SOLVEPNP_AP3P;
    _minimalRequired = 4;

    _blurMethod = _BOX_BLUR;
    _blurKernelSize = 6;

    _downSample = 0.5;
    _convertToGrayscale = false;

    _numKDtrees = 1;
    _numTables = 10;
    _topK = 5;
    _ratioThreshold_Sig2Q = 1.0;
    _ratioThreshold_Q2Sig = 0.9;
    _maximumNNL2Dist = INFINITY;
    _qFeaturesLimit = 0;
    _matchMode = _MATCH_MODE_MUTUAL;

    _maximumNNHammingDist = numeric_limits<unsigned int>::max();
    _poseEstMethod = _POSE_EST_BY_VOTE;
    _refineEstimate = false;

    _renderBlurred = false;
    _renderFeatures = false;
    _renderInliers = false;
    _writeDetected = false;
    _writeObjCorr = false;
    _writeObjInliers = false;
    _verbose = false;
    _helpme = false;

    briskConfigFPlen = 0;
    orbConfigFPlen = 0;
    siftConfigFPlen = 0;
    surfConfigFPlen = 0;

    _nonMaxSuppression = 0.0;                                       //  No suppression

    _brisk_threshold = 30;                                          //  BRISK defaults
    _brisk_octaves = 3;
    _brisk_patternScale = 1.0;

    _orb_scaleFactor = 1.2;                                         //  ORB defaults
    _orb_levels = 8;
    _orb_edgeThreshold = 31;
    _orb_firstLevel = 0;
    _orb_wta_k = 2;
    _orb_scoreType = cv::ORB::HARRIS_SCORE;
    _orb_patchSize = 31;
    _orb_fastThreshold = 20;

    _sift_octaveLayers = 3;                                         //  SIFT defaults
    _sift_contrastThreshold = 0.04;
    _sift_edgeThreshold = 10.0;
    _sift_sigma = 1.6;

    _surf_hessianThreshold = 100.0;                                 //  SURF defaults
    _surf_octaves = 4;
    _surf_octaveLayers = 3;
  }

/* PnPConfig constructor, given argc and argv */
PnPConfig::PnPConfig(int argc, char** argv)
  {
    #ifdef __PNP_CONFIG_DEBUG
    cout << "PnPConfig::PnPConfig(" << +argc << ", argv)" << endl;
    #endif

    QFPlen = 0;
    SigFPlen = 0;
    KFNlen = 0;
    RtFNlen = 0;
    distortionFNlen = 0;
    SigFPIsDirectory = false;
    maskFPlen = 0;

    _iterations = 2000;
    _reprojectionError = 2.0;
    _confidence = 0.9999;
    _ransacMethod = SOLVEPNP_AP3P;
    _minimalRequired = 4;

    _blurMethod = _BOX_BLUR;
    _blurKernelSize = 6;

    _downSample = 0.5;
    _convertToGrayscale = false;

    _numKDtrees = 1;
    _numTables = 10;
    _topK = 5;
    _ratioThreshold_Sig2Q = 1.0;
    _ratioThreshold_Q2Sig = 0.9;
    _maximumNNL2Dist = INFINITY;
    _qFeaturesLimit = 0;
    _matchMode = _MATCH_MODE_MUTUAL;

    _maximumNNHammingDist = numeric_limits<unsigned int>::max();
    _poseEstMethod = _POSE_EST_BY_VOTE;
    _refineEstimate = false;

    _renderBlurred = false;
    _renderFeatures = false;
    _renderInliers = false;
    _writeDetected = false;
    _writeObjCorr = false;
    _writeObjInliers = false;
    _verbose = false;
    _helpme = false;

    briskConfigFPlen = 0;
    orbConfigFPlen = 0;
    siftConfigFPlen = 0;
    surfConfigFPlen = 0;

    _nonMaxSuppression = 0.0;                                       //  No suppression

    _brisk_threshold = 30;                                          //  BRISK defaults
    _brisk_octaves = 3;
    _brisk_patternScale = 1.0;

    _orb_scaleFactor = 1.2;                                         //  ORB defaults
    _orb_levels = 8;
    _orb_edgeThreshold = 31;
    _orb_firstLevel = 0;
    _orb_wta_k = 2;
    _orb_scoreType = cv::ORB::HARRIS_SCORE;
    _orb_patchSize = 31;
    _orb_fastThreshold = 20;

    _sift_octaveLayers = 3;                                         //  SIFT defaults
    _sift_contrastThreshold = 0.04;
    _sift_edgeThreshold = 10.0;
    _sift_sigma = 1.6;

    _surf_hessianThreshold = 100.0;                                 //  SURF defaults
    _surf_octaves = 4;
    _surf_octaveLayers = 3;

    parse(argc, argv);
  }

/**************************************************************************************************
 Destructor  */

PnPConfig::~PnPConfig()
  {
  }

/**************************************************************************************************
 Signatures  */

/* Write signature file names to 'charBuffer', file name lengths to 'lengthBuffer';
  return signature count */
unsigned int PnPConfig::fetchSignatures(char** charBuffer, unsigned int** lengthBuffer) const
  {
    unsigned int numSig = 0;
    unsigned int totalLen = 0;
    char buffer[1024];

    DIR *dir;                                                       //  Open the signatures directory
    struct dirent *ent;                                             //  Directory entry
    struct stat st;                                                 //  Determine file or directory

    unsigned int i, j;

    if(SigFPlen > 0)
      {
        if(SigFPIsDirectory)                                        //  Potentially several signature files
          {
            dir = opendir(SigFP);                                   //  Open target-object directory and read all files
            while((ent = readdir(dir)) != NULL)
              {
                if(SigFP[SigFPlen - 1] == '/')                      //  Did the given signature path include a trailing slash?
                  i = sprintf(buffer, "%s%s", SigFP, ent->d_name);
                else
                  i = sprintf(buffer, "%s/%s", SigFP, ent->d_name);
                buffer[i] = '\0';                                   //  Null-cap the file path
                stat(buffer, &st);
                                                                    //  Ignore current and parent directories
                if(strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0 && st.st_mode & S_IFREG)
                  {
                    numSig++;                                       //  Count up signatures
                    totalLen += i + 1;                              //  Add up string lengths
                  }
              }
            closedir(dir);                                          //  Done reading object directory

            if(((*charBuffer) = (char*)malloc((totalLen + 1) * sizeof(char))) == NULL)
              {
                #ifdef __PNP_CONFIG_DEBUG
                cout << "ERROR: Unable to allocate signature file name buffer." << endl;
                #endif
                return 0;
              }
            if(((*lengthBuffer) = (unsigned int*)malloc(numSig * sizeof(int))) == NULL)
              {
                #ifdef __PNP_CONFIG_DEBUG
                cout << "ERROR: Unable to allocate signature file-name length buffer." << endl;
                #endif
                free((*charBuffer));
                return 0;
              }

            numSig = 0;                                             //  Reset
            totalLen = 0;                                           //  Reset

            dir = opendir(SigFP);                                   //  Open target-object directory and read all files
            while((ent = readdir(dir)) != NULL)
              {
                if(SigFP[SigFPlen - 1] == '/')                      //  Did the given signature path include a trailing slash?
                  i = sprintf(buffer, "%s%s", SigFP, ent->d_name);
                else
                  i = sprintf(buffer, "%s/%s", SigFP, ent->d_name);
                buffer[i] = '\0';                                   //  Null-cap the file path
                stat(buffer, &st);
                                                                    //  Ignore current and parent directories
                if(strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0 && st.st_mode & S_IFREG)
                  {
                    (*lengthBuffer)[numSig] = i;
                    numSig++;

                    for(j = 0; j < i; j++)
                      (*charBuffer)[totalLen + j] = buffer[j];
                    (*charBuffer)[totalLen + j] = '\0';
                    totalLen += i + 1;                              //  Add up string lengths
                  }
              }
            closedir(dir);                                          //  Done reading object directory
          }
        else                                                        //  A single signature file
          {
            if(((*charBuffer) = (char*)malloc((SigFPlen + 1) * sizeof(char))) == NULL)
              {
                #ifdef __PNP_CONFIG_DEBUG
                cout << "ERROR: Unable to allocate signature file name buffer." << endl;
                #endif
                return 0;
              }
            if(((*lengthBuffer) = (unsigned int*)malloc(sizeof(int))) == NULL)
              {
                #ifdef __PNP_CONFIG_DEBUG
                cout << "ERROR: Unable to allocate signature file-name length buffer." << endl;
                #endif
                free((*charBuffer));
                return 0;
              }

            numSig = 1;

            for(i = 0; i < SigFPlen; i++)
              (*charBuffer)[i] = SigFP[i];
            (*charBuffer)[i] = '\0';

            (*lengthBuffer)[0] = SigFPlen;
          }
      }

    return numSig;
  }

/**************************************************************************************************
 Parser  */

/* PnPConfig constructor, no data given */
bool PnPConfig::parse(int argc, char** argv)
  {
    #ifdef __PNP_CONFIG_DEBUG
    cout << "PnPConfig::parse(" << +argc << ", argv)" << endl;
    #endif

    unsigned int i = 1;
    unsigned int j;
    char buffer[512];
    unsigned char argtarget = PARAM_NONE;

    while(i < (unsigned int)argc)
      {
        if(strcmp(argv[i], "-v") == 0)                              //  Enable verbosity
          {
            _verbose = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-Q") == 0)                         //  String to follow is the file path for a query image
          argtarget = PARAM_Q;
        else if(strcmp(argv[i], "-sig") == 0)                       //  String to follow is the file path for the signature(s)
          argtarget = PARAM_SIG;
        else if(strcmp(argv[i], "-K") == 0)                         //  String to follow is the name of the intrinsic file to use
          argtarget = PARAM_K;
        else if(strcmp(argv[i], "-Rt") == 0)                        //  String to follow is the name of the extrinsic file to use
          argtarget = PARAM_RT;
        else if(strcmp(argv[i], "-dist") == 0)                      //  String to follow is the name of the distortion file to use
          argtarget = PARAM_DISTORT;
        else if(strcmp(argv[i], "-kd") == 0)                        //  Following integer is the number of KD-trees to use in index
          argtarget = PARAM_KDTREE;
        else if(strcmp(argv[i], "-lsh") == 0)                       //  Following integer is the number of LSH hash tables
          argtarget = PARAM_HASHTABLES;
        else if(strcmp(argv[i], "-iter") == 0)                      //  Following integer is the number of iteration for RANSAC to run
          argtarget = PARAM_ITER;
        else if(strcmp(argv[i], "-rErr") == 0)                      //  Following float is the toerable reprojection error
          argtarget = PARAM_REPROJ_ERR;
        else if(strcmp(argv[i], "-conf") == 0)                      //  Following float is the target pose confidence
          argtarget = PARAM_CONF;
        else if(strcmp(argv[i], "-meth") == 0)                      //  Following string sets the RANSAC method
          argtarget = PARAM_METHOD;
        else if(strcmp(argv[i], "-bmeth") == 0)                     //  Following string sets the blur method
          argtarget = PARAM_BLUR_TYPE;
        else if(strcmp(argv[i], "-bkernel") == 0)                   //  Following string sets the blur kernel size
          argtarget = PARAM_BLUR_K_SIZE;
        else if(strcmp(argv[i], "-down") == 0)                      //  Following float is the down-sampling factor
          argtarget = PARAM_DOWNSAMPLE;
        else if(strcmp(argv[i], "-top") == 0)                       //  Following integer sets number of top candidates to consider
          argtarget = PARAM_TOP_K;
        else if(strcmp(argv[i], "-ratio") == 0)                     //  Following float is the nearest neighbor ratio threshold in BOTH DIRECTIONS
          argtarget = PARAM_RATIO_THRESH;
        else if(strcmp(argv[i], "-ratioSig2Q") == 0)                //  Following float is the nearest neighbor ratio threshold for SIG-->Q
          argtarget = PARAM_RATIO_SIG2Q;
        else if(strcmp(argv[i], "-ratioQ2Sig") == 0)                //  Following float is the nearest neighbor ratio threshold for Q-->SIG
          argtarget = PARAM_RATIO_Q2SIG;
        else if(strcmp(argv[i], "-maxNNL2") == 0)                   //  Following float is the maximum nearest neighbor distance
          argtarget = PARAM_MAX_NN_L2_DIST;
        else if(strcmp(argv[i], "-maxNNHamm") == 0)                 //  Following uint is the maximum nearest neighbor Hamming distance
          argtarget = PARAM_MAX_NN_HAMM_DIST;
        else if(strcmp(argv[i], "-maxQ") == 0)                      //  Following uint is the maximum number of features to extract from the query
          argtarget = PARAM_Q_LIMIT_FEATURES;
        else if(strcmp(argv[i], "-nms") == 0)                       //  Following uint is the radius in pixels around which to suppress lesser features
          argtarget = PARAM_NONMAXSUPPR;
        else if(strcmp(argv[i], "-mmeth") == 0)                     //  Following string indicates the matching method
          argtarget = PARAM_MATCH_MODE;
        else if(strcmp(argv[i], "-refine") == 0)                    //  Refine pose estimate by re-running RANSAC on inliers
          {
            _refineEstimate = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-gray") == 0)                      //  Convert query image to grayscale before detecting features
          {
            _convertToGrayscale = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-showBlur") == 0)                  //  Render the blurred image
          {
            _renderBlurred = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-showFeatures") == 0)              //  Render the features to image
          {
            _renderFeatures = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-showInliers") == 0)               //  Render the inliers to image
          {
            _renderInliers = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-outFeat") == 0)                   //  Write detected points to text file
          {
            _writeDetected = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-outCorr") == 0)                   //  Write object's pre-RANSAC correspondences to text file
          {
            _writeObjCorr = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-outInlier") == 0)                 //  Write object's inlier-correspondences to text file
          {
            _writeObjInliers = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-pemeth") == 0)                    //  Following string sets the pose estimation method
          argtarget = PARAM_POSE_EST_METHOD;
                                                                    //  Show usage and halt
        else if(strcmp(argv[i], "-?") == 0 || strcmp(argv[i], "-help") == 0 ||strcmp(argv[i], "--help") == 0)
          {
            _helpme = true;
            argtarget = PARAM_NONE;
          }
        else if(strcmp(argv[i], "-BRISKcfg") == 0)                  //  String to follow is the file path for a BRISK config file
          argtarget = PARAM_BRISK_CONFIG;
        else if(strcmp(argv[i], "-ORBcfg") == 0)                    //  String to follow is the file path for an ORB config file
          argtarget = PARAM_ORB_CONFIG;
        else if(strcmp(argv[i], "-SIFTcfg") == 0)                   //  String to follow is the file path for a SIFT config file
          argtarget = PARAM_SIFT_CONFIG;
        else if(strcmp(argv[i], "-SURfcfg") == 0)                   //  String to follow is the file path for a SURF config file
          argtarget = PARAM_SURF_CONFIG;
        else if(strcmp(argv[i], "-mask") == 0)                      //  String to follow is the file path for an Extractor mask
          argtarget = PARAM_MASK;
        else                                                        //  Not one of our flags... react to one of the flags
          {
            switch(argtarget)
              {
                case PARAM_Q:
                  QFPlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((QFP = (char*)malloc((QFPlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < QFPlen; j++)
                    QFP[j] = buffer[j];
                  QFP[j] = '\0';

                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_SIG:                                     //  Incoming signature(s) file path
                  SigFPlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((SigFP = (char*)malloc((SigFPlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < SigFPlen; j++)
                    SigFP[j] = buffer[j];
                  SigFP[j] = '\0';

                  SigFPIsDirectory = isSignaturePathDirectory();

                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_K:                                       //  Incoming intrinsics file name
                  KFNlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((KFN = (char*)malloc((KFNlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < KFNlen; j++)
                    KFN[j] = buffer[j];
                  KFN[j] = '\0';
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_RT:                                      //  Incoming extrinsics file name
                  RtFNlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((RtFN = (char*)malloc((RtFNlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < RtFNlen; j++)
                    RtFN[j] = buffer[j];
                  RtFN[j] = '\0';
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_DISTORT:                                 //  Incoming distortion coefficients file name
                  distortionFNlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((distortionFN = (char*)malloc((distortionFNlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < distortionFNlen; j++)
                    distortionFN[j] = buffer[j];
                  distortionFN[j] = '\0';
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_KDTREE:                                  //  Incoming unsigned integer
                  _numKDtrees = (unsigned int)atoi(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_HASHTABLES:                              //  Incoming unsigned integer
                  _numTables = (unsigned int)atoi(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_ITER:                                    //  Incoming unsigned integer
                  _iterations = (unsigned int)atoi(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_REPROJ_ERR:                              //  Incoming float
                  _reprojectionError = (float)atof(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_CONF:                                    //  Incoming float
                  _confidence = (float)atof(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_MATCH_MODE:                              //  Incoming string
                  if(strcmp(argv[i], "SIG2Q") == 0)
                    _matchMode = _MATCH_MODE_SIG2Q;
                  else if(strcmp(argv[i], "Q2SIG") == 0)
                    _matchMode = _MATCH_MODE_Q2SIG;
                  else if(strcmp(argv[i], "MUT") == 0)
                    _matchMode = _MATCH_MODE_MUTUAL;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_RATIO_THRESH:                            //  Incoming float
                  _ratioThreshold_Sig2Q = (float)atof(argv[i]);
                  if(_ratioThreshold_Sig2Q < MINIMUM_RATIO_THRESHOLD)
                    _ratioThreshold_Sig2Q = MINIMUM_RATIO_THRESHOLD;
                  else if(_ratioThreshold_Sig2Q > 1.0)
                    _ratioThreshold_Sig2Q = 1.0;
                  _ratioThreshold_Q2Sig = _ratioThreshold_Sig2Q;    //  Set for both directions
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_RATIO_SIG2Q:                             //  Incoming float
                  _ratioThreshold_Sig2Q = (float)atof(argv[i]);
                  if(_ratioThreshold_Sig2Q < MINIMUM_RATIO_THRESHOLD)
                    _ratioThreshold_Sig2Q = MINIMUM_RATIO_THRESHOLD;
                  else if(_ratioThreshold_Sig2Q > 1.0)
                    _ratioThreshold_Sig2Q = 1.0;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_RATIO_Q2SIG:                             //  Incoming float
                  _ratioThreshold_Q2Sig = (float)atof(argv[i]);
                  if(_ratioThreshold_Q2Sig < MINIMUM_RATIO_THRESHOLD)
                    _ratioThreshold_Q2Sig = MINIMUM_RATIO_THRESHOLD;
                  else if(_ratioThreshold_Q2Sig > 1.0)
                    _ratioThreshold_Q2Sig = 1.0;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_MAX_NN_L2_DIST:                          //  Incoming float
                  if(strcmp(argv[i], "inf") == 0)                   //  INFINIFY --> no restrictions on nearness
                    _maximumNNL2Dist = INFINITY;
                  else
                    _maximumNNL2Dist = fabs((float)atof(argv[i]));
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_MAX_NN_HAMM_DIST:                        //  Incoming uint
                  if(strcmp(argv[i], "inf") == 0)                   //  MAX_INT --> no restriction on nearness
                    _maximumNNHammingDist = std::numeric_limits<unsigned int>::max();
                  else
                    _maximumNNHammingDist = abs((unsigned int)atoi(argv[i]));
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_METHOD:                                  //  Incoming string sets integer
                  if(strcmp(argv[i], "P3P") == 0)
                    _ransacMethod = SOLVEPNP_P3P;
                  else if(strcmp(argv[i], "EPNP") == 0)
                    _ransacMethod = SOLVEPNP_EPNP;
                  else if(strcmp(argv[i], "ITERATIVE") == 0)
                    _ransacMethod = SOLVEPNP_ITERATIVE;
                  else if(strcmp(argv[i], "DLS") == 0)
                    _ransacMethod = SOLVEPNP_DLS;
                  else if(strcmp(argv[i], "UPNP") == 0)
                    _ransacMethod = SOLVEPNP_UPNP;
                  else if(strcmp(argv[i], "AP3P") == 0)
                    _ransacMethod = SOLVEPNP_AP3P;
                  else if(strcmp(argv[i], "IPPE") == 0)
                    _ransacMethod = SOLVEPNP_IPPE;
                  else if(strcmp(argv[i], "IPPESQ") == 0)
                    _ransacMethod = SOLVEPNP_IPPE_SQUARE;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_BLUR_TYPE:                               //  Incoming string
                  if(strcmp(argv[i], "BOX") == 0)
                    _blurMethod = _BOX_BLUR;
                  else if(strcmp(argv[i], "GAUSS") == 0)
                    _blurMethod = _GAUSSIAN_BLUR;
                  else if(strcmp(argv[i], "MED") == 0)
                    _blurMethod = _MEDIAN_BLUR;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_BLUR_K_SIZE:                             //  Incoming integer
                  _blurKernelSize = (unsigned int)atoi(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_DOWNSAMPLE:                              //  Incoming float
                  _downSample = (float)atof(argv[i]);
                  if(_downSample > 1.0 || _downSample <= 0.0)       //  Don't take no nonsense
                    _downSample = 1.0;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_TOP_K:                                   //  Incoming integer
                  _topK = (unsigned int)atoi(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_POSE_EST_METHOD:                         //  Incoming string
                  if(strcmp(argv[i], "VOTES") == 0)
                    _poseEstMethod = _POSE_EST_BY_VOTE;
                  else if(strcmp(argv[i], "INDEP") == 0)
                    _poseEstMethod = _POSE_EST_INDEPENDENT;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_Q_LIMIT_FEATURES:                        //  Incoming integer
                  _qFeaturesLimit = (unsigned int)atoi(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_BRISK_CONFIG:                            //  Incoming string
                  briskConfigFPlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((briskConfigFP = (char*)malloc((briskConfigFPlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < briskConfigFPlen; j++)
                    briskConfigFP[j] = buffer[j];
                  briskConfigFP[j] = '\0';

                  loadBRISKparams();                                //  Load the settings immediately

                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_ORB_CONFIG:                              //  Incoming string
                  orbConfigFPlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((orbConfigFP = (char*)malloc((orbConfigFPlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < orbConfigFPlen; j++)
                    orbConfigFP[j] = buffer[j];
                  orbConfigFP[j] = '\0';

                  loadORBparams();                                  //  Load the settings immediately

                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_SIFT_CONFIG:                             //  Incoming string
                  siftConfigFPlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((siftConfigFP = (char*)malloc((siftConfigFPlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < siftConfigFPlen; j++)
                    siftConfigFP[j] = buffer[j];
                  siftConfigFP[j] = '\0';

                  loadSIFTparams();                                 //  Load the settings immediately

                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_SURF_CONFIG:                             //  Incoming string
                  surfConfigFPlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((surfConfigFP = (char*)malloc((surfConfigFPlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < surfConfigFPlen; j++)
                    surfConfigFP[j] = buffer[j];
                  surfConfigFP[j] = '\0';

                  loadSURFparams();                                 //  Load the settings immediately

                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_MASK:                                    //  Incoming string
                  maskFPlen = (unsigned int)sprintf(buffer, "%s", argv[i]);
                  if((maskFP = (char*)malloc((maskFPlen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < maskFPlen; j++)
                    maskFP[j] = buffer[j];
                  maskFP[j] = '\0';

                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_NONMAXSUPPR:                             //  Incoming float
                  _nonMaxSuppression = (float)atof(argv[i]);
                  if(_nonMaxSuppression < 0.0)                      //  Force non-negative
                    _nonMaxSuppression *= -1.0;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;
              }
          }

        i++;
      }

    return true;
  }

/**************************************************************************************************
 Getters  */

unsigned int PnPConfig::Q(char** buffer) const
  {
    unsigned int i;
    if(((*buffer) = (char*)malloc((QFPlen + 1) * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate given buffer for query image file path." << endl;
        return 0;
      }
    for(i = 0; i < QFPlen; i++)
      (*buffer)[i] = QFP[i];
    (*buffer)[i] = '\0';
    return QFPlen;
  }

unsigned int PnPConfig::Sig(char** buffer) const
  {
    unsigned int i;
    if(((*buffer) = (char*)malloc((SigFPlen + 1) * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate given buffer for signature file path." << endl;
        return 0;
      }
    for(i = 0; i < SigFPlen; i++)
      (*buffer)[i] = SigFP[i];
    (*buffer)[i] = '\0';
    return SigFPlen;
  }

unsigned int PnPConfig::K(char** buffer) const
  {
    unsigned int i;
    if(((*buffer) = (char*)malloc((KFNlen + 1) * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate given buffer for K file name." << endl;
        return 0;
      }
    for(i = 0; i < KFNlen; i++)
      (*buffer)[i] = KFN[i];
    (*buffer)[i] = '\0';
    return KFNlen;
  }

unsigned int PnPConfig::Rt(char** buffer) const
  {
    unsigned int i;
    if(((*buffer) = (char*)malloc((RtFNlen + 1) * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate given buffer for Rt file name." << endl;
        return 0;
      }
    for(i = 0; i < RtFNlen; i++)
      (*buffer)[i] = RtFN[i];
    (*buffer)[i] = '\0';
    return RtFNlen;
  }

unsigned int PnPConfig::distortion(char** buffer) const
  {
    unsigned int i;
    if(((*buffer) = (char*)malloc((distortionFNlen + 1) * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate given buffer for distortion-coefficient file name." << endl;
        return 0;
      }
    for(i = 0; i < distortionFNlen; i++)
      (*buffer)[i] = distortionFN[i];
    (*buffer)[i] = '\0';
    return distortionFNlen;
  }

unsigned int PnPConfig::mask(char** buffer) const
  {
    unsigned int i;
    if(((*buffer) = (char*)malloc((maskFPlen + 1) * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate given buffer for distortion-coefficient file name." << endl;
        return 0;
      }
    for(i = 0; i < maskFPlen; i++)
      (*buffer)[i] = maskFP[i];
    (*buffer)[i] = '\0';
    return maskFPlen;
  }

bool PnPConfig::hasMask() const
  {
    return (maskFPlen > 0);
  }

unsigned int PnPConfig::iterations() const
  {
    return _iterations;
  }

float PnPConfig::reprojectionError() const
  {
    return _reprojectionError;
  }

float PnPConfig::confidence() const
  {
    return _confidence;
  }

int PnPConfig::ransacMethod() const
  {
    return _ransacMethod;
  }

unsigned char PnPConfig::minimalRequired() const
  {
    return _minimalRequired;
  }

unsigned int PnPConfig::qLimit() const
  {
    return _qFeaturesLimit;
  }

unsigned char PnPConfig::blurMethod() const
  {
    return _blurMethod;
  }

unsigned char PnPConfig::blurKernelSize() const
  {
    return _blurKernelSize;
  }

float PnPConfig::downSample() const
  {
    return _downSample;
  }

bool PnPConfig::convertToGrayscale() const
  {
    return _convertToGrayscale;
  }

float PnPConfig::nonMaxSuppression() const
  {
    return _nonMaxSuppression;
  }

unsigned int PnPConfig::numKDtrees() const
  {
    return _numKDtrees;
  }

unsigned int PnPConfig::numTables() const
  {
    return _numTables;
  }

unsigned int PnPConfig::topK() const
  {
    return _topK;
  }

unsigned char PnPConfig::matchMethod() const
  {
    return _matchMode;
  }

float PnPConfig::ratioThresholdSig2Q() const
  {
    return _ratioThreshold_Sig2Q;
  }

float PnPConfig::ratioThresholdQ2Sig() const
  {
    return _ratioThreshold_Q2Sig;
  }

float PnPConfig::maximumNNL2Dist() const
  {
    return _maximumNNL2Dist;
  }

unsigned int PnPConfig::maximumNNHammingDist() const
  {
    return _maximumNNHammingDist;
  }

unsigned char PnPConfig::poseEstMethod() const
  {
    return _poseEstMethod;
  }

bool PnPConfig::refineEstimate() const
  {
    return _refineEstimate;
  }

int PnPConfig::brisk_threshold() const
  {
    return _brisk_threshold;
  }

int PnPConfig::brisk_octaves() const
  {
    return _brisk_octaves;
  }

float PnPConfig::brisk_patternScale() const
  {
    return _brisk_patternScale;
  }

float PnPConfig::orb_scaleFactor() const
  {
    return _orb_scaleFactor;
  }

int PnPConfig::orb_levels() const
  {
    return _orb_levels;
  }

int PnPConfig::orb_edgeThreshold() const
  {
    return _orb_edgeThreshold;
  }

int PnPConfig::orb_firstLevel() const
  {
    return _orb_firstLevel;
  }

int PnPConfig::orb_wta_k() const
  {
    return _orb_wta_k;
  }

int PnPConfig::orb_scoreType() const
  {
    return _orb_scoreType;
  }

int PnPConfig::orb_patchSize() const
  {
    return _orb_patchSize;
  }

int PnPConfig::orb_fastThreshold() const
  {
    return _orb_fastThreshold;
  }

int PnPConfig::sift_octaveLayers() const
  {
    return _sift_octaveLayers;
  }

double PnPConfig::sift_contrastThreshold() const
  {
    return _sift_contrastThreshold;
  }

double PnPConfig::sift_edgeThreshold() const
  {
    return _sift_edgeThreshold;
  }

double PnPConfig::sift_sigma() const
  {
    return _sift_sigma;
  }

double PnPConfig::surf_hessianThreshold() const
  {
    return _surf_hessianThreshold;
  }

int PnPConfig::surf_octaves() const
  {
    return _surf_octaves;
  }

int PnPConfig::surf_octaveLayers() const
  {
    return _surf_octaveLayers;
  }

bool PnPConfig::renderBlurred() const
  {
    return _renderBlurred;
  }

bool PnPConfig::renderFeatures() const
  {
    return _renderFeatures;
  }

bool PnPConfig::renderInliers() const
  {
    return _renderInliers;
  }

bool PnPConfig::writeDetected() const
  {
    return _writeDetected;
  }

bool PnPConfig::writeObjCorr() const
  {
    return _writeObjCorr;
  }

bool PnPConfig::writeObjInliers() const
  {
    return _writeObjInliers;
  }

bool PnPConfig::verbose() const
  {
    return _verbose;
  }

bool PnPConfig::helpme() const
  {
    return _helpme;
  }

bool PnPConfig::isSignaturePathDirectory() const
  {
    struct stat st;                                                 //  Determine file or directory

    if(SigFPlen > 0)
      {
        if(stat(SigFP, &st) == 0)
          {
            if(st.st_mode & S_IFDIR)                                //  The signature argument is a directory
              return true;
            if(st.st_mode & S_IFREG)                                //  The signature argument is a file
              return false;
          }
      }

    return false;
  }

/**************************************************************************************************
 Load detector parameters from file  */

/* BRISK configuration = [int, int, float] */
bool PnPConfig::loadBRISKparams()
  {
    FILE* fp;
    float tmpF;
    int tmpI;
    bool good;

    #ifdef __PNP_CONFIG_DEBUG
    cout << "PnPConfig::loadBRISKparams(" << briskConfigFP << ")" << endl;
    #endif

    if((fp = fopen(briskConfigFP, "rb")) == NULL)                   //  Open file for reading
      {
        cout << "ERROR: Unable to open \"" << briskConfigFP << "\"." << endl;
        return false;
      }

    fseek(fp, 0, SEEK_SET);                                         //  Start at the beginning

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read BRISK's threshold
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _brisk_threshold = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read BRISK's number of octaves
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _brisk_octaves = tmpI;

    good = ((fread(&tmpF, sizeof(float), 1, fp)) == 1);             //  Read BRISK's pattern scale
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _brisk_patternScale = tmpF;

    fclose(fp);
    return true;
  }

/* ORB configuration = [float, int, int, int, int, int, int, int] */
bool PnPConfig::loadORBparams()
  {
    FILE* fp;
    float tmpF;
    int tmpI;
    bool good;

    #ifdef __PNP_CONFIG_DEBUG
    cout << "PnPConfig::loadORBparams(" << orbConfigFP << ")" << endl;
    #endif

    if((fp = fopen(orbConfigFP, "rb")) == NULL)                     //  Open file for reading
      {
        cout << "ERROR: Unable to open \"" << orbConfigFP << "\"." << endl;
        return false;
      }

    fseek(fp, 0, SEEK_SET);                                         //  Start at the beginning

    good = ((fread(&tmpF, sizeof(float), 1, fp)) == 1);             //  Read ORB's scale factor
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_scaleFactor = tmpF;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read ORB's number of levels
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_levels = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read ORB's threshold
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_edgeThreshold = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read ORB's first level
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_firstLevel = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read ORB's WTA_K
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_wta_k = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read ORB's score type
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_scoreType = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read ORB's patch size
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_patchSize = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read ORB's FAST threshold
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _orb_fastThreshold = tmpI;

    fclose(fp);
    return true;
  }

/* SIFT configuration = [int, double, double, double] */
bool PnPConfig::loadSIFTparams()
  {
    FILE* fp;
    double tmpF;
    int tmpI;
    bool good;

    #ifdef __PNP_CONFIG_DEBUG
    cout << "PnPConfig::loadSIFTparams(" << siftConfigFP << ")" << endl;
    #endif

    if((fp = fopen(siftConfigFP, "rb")) == NULL)                    //  Open file for reading
      {
        cout << "ERROR: Unable to open \"" << siftConfigFP << "\"." << endl;
        return false;
      }

    fseek(fp, 0, SEEK_SET);                                         //  Start at the beginning

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read SIFT's number of octaves
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _sift_octaveLayers = tmpI;

    good = ((fread(&tmpF, sizeof(double), 1, fp)) == 1);            //  Read SIFT's contrast threshold
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _sift_contrastThreshold = tmpF;

    good = ((fread(&tmpF, sizeof(double), 1, fp)) == 1);            //  Read SIFT's edge threshold
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _sift_edgeThreshold = tmpF;

    good = ((fread(&tmpF, sizeof(double), 1, fp)) == 1);            //  Read SIFT's Gaussian sigma
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _sift_sigma = tmpF;

    fclose(fp);
    return true;
  }

/* SURF configuration = [double, int, int] */
bool PnPConfig::loadSURFparams()
  {
    FILE* fp;
    double tmpF;
    int tmpI;
    bool good;

    #ifdef __PNP_CONFIG_DEBUG
    cout << "PnPConfig::loadSURFparams(" << surfConfigFP << ")" << endl;
    #endif

    if((fp = fopen(surfConfigFP, "rb")) == NULL)                    //  Open file for reading
      {
        cout << "ERROR: Unable to open \"" << surfConfigFP << "\"." << endl;
        return false;
      }

    fseek(fp, 0, SEEK_SET);                                         //  Start at the beginning

    good = ((fread(&tmpF, sizeof(double), 1, fp)) == 1);            //  Read SURF's Hessian threshold
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _surf_hessianThreshold = tmpF;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read number of SURF's pyramid octaves
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _surf_octaves = tmpI;

    good = ((fread(&tmpI, sizeof(int), 1, fp)) == 1);               //  Read number of SURF's layers per octave
    if(!good)
      {
        fclose(fp);
        return false;
      }
    _surf_octaveLayers = tmpI;

    fclose(fp);
    return true;
  }

/**************************************************************************************************
 Summary  */

void PnPConfig::display() const
  {
    cout << "Query:                                       ";        //  Show me the query-image filepath
    if(QFPlen > 0)
      cout << QFP << endl;
    else
      cout << "<NONE>" << endl;

    cout << "Signatures:                                  ";        //  Show me the signatures-filepath
    if(SigFPlen > 0)
      cout << SigFP << endl;
    else
      cout << "<NONE>" << endl;

    cout << "K-file:                                      ";        //  Show me the K-file
    if(KFNlen > 0)
      cout << KFN << endl;
    else
      cout << "<NONE>" << endl;

    cout << "Rt-file:                                     ";        //  Show me the Rt-file
    if(RtFNlen > 0)
      cout << RtFN << endl;
    else
      cout << "<NONE>" << endl;

    cout << "Distortion-file:                             ";        //  Show me the Distortion-coefficients-file
    if(distortionFNlen > 0)
      cout << distortionFN << endl;
    else
      cout << "<NONE>" << endl;
                                                                    //  Show me the number of KD-Trees in an L2-Index
    cout << "Number of KD-Trees:                          " << +_numKDtrees << endl;
                                                                    //  Show me the number of hash tables in a Hamming-Index
    cout << "Number of Locality-Sensitive Hash tables:    " << +_numTables << endl;
    cout << "Pose-estimation method:                      ";
    switch(_poseEstMethod)
      {
        case _POSE_EST_BY_VOTE:      cout << "VOTE" << endl;  break;
        case _POSE_EST_INDEPENDENT:  cout << "INDEP" << endl;  break;
      }
    if(_matchMode == _MATCH_MODE_SIG2Q)                             //  Show me the matching direction
      cout << "Index(Signatures), Query(image)." << endl;
    else if(_matchMode == _MATCH_MODE_Q2SIG)
      cout << "Index(image), Query(Signatures)." << endl;
    else
      cout << "Index(Signatures), Query(image) + Index(image), Query(Signatures)." << endl;
                                                                    //  Show me the number of RANSAC iterations
    cout << "Number of RANSAC iterations:                 " << +_iterations << endl;
                                                                    //  Show me the reprojection error
    cout << "Reprojection error:                          " << _reprojectionError << endl;
                                                                    //  Show me the number of top candidates to consider
    cout << "Top candidates:                              " << +_topK << endl;
                                                                    //  Show me the confidence threshold
    cout << "Confidence:                                  " << _confidence << endl;
                                                                    //  Show me the ratio thresholds
    cout << "Lowe-ratio, Sig-->Q:                         " << _ratioThreshold_Sig2Q << endl;
    cout << "Lowe-ratio, Q-->Sig:                         " << _ratioThreshold_Q2Sig << endl;
                                                                    //  Show me the greatest distance a NN can be and still be an L2 neighbor
    cout << "Maximum Nearest-Neighbor L2 distance:        " << _maximumNNL2Dist << endl;
                                                                    //  Show me the greatest distance a NN can be and still be an L2 neighbor
    if(_maximumNNHammingDist < numeric_limits<unsigned int>::max())
      cout << "Maximum Nearest-Neighbor Hamming distance:   " << +_maximumNNHammingDist << endl;
    else
      cout << "Maximum Nearest-Neighbor Hamming distance:   inf" << endl;
                                                                    //  Show me the maximum number of query features
    cout << "Maximum number of query-features to extract: " << +_qFeaturesLimit << endl;
    if(_nonMaxSuppression > 0.0)
      cout << "Non-maximum suppression radius in pixels:    " << +_nonMaxSuppression << endl;
    else
      cout << "No features will be suppressed." << endl;
    cout << "PnP-solver method:                           ";        //  Show me the PnP-solver method
    switch(_ransacMethod)
      {
        case SOLVEPNP_P3P:          cout << "P3P";        break;
        case SOLVEPNP_EPNP:         cout << "EPNP";       break;
        case SOLVEPNP_ITERATIVE:    cout << "ITERATIVE";  break;
        case SOLVEPNP_DLS:          cout << "DLS";        break;
        case SOLVEPNP_UPNP:         cout << "UPNP";       break;
        case SOLVEPNP_AP3P:         cout << "AP3P";       break;
        case SOLVEPNP_IPPE:         cout << "IPPE";       break;
        case SOLVEPNP_IPPE_SQUARE:  cout << "IPPESQ";     break;
      }
    cout << endl;
    cout << "Blur method:                                 ";        //  Show me the blur kernel method
    switch(_blurMethod)
      {
        case _BOX_BLUR:       cout << "BOX";  break;
        case _GAUSSIAN_BLUR:  cout << "GAUSSIAN";  break;
        case _MEDIAN_BLUR:    cout << "MEDIAN";  break;
      }
    cout << endl;
                                                                    //  Show me the blur kernel method
    cout << "Blur kernel size:                            " << +_blurKernelSize << endl;
    if(_downSample == 1.0)
      cout << "No down-sampling" << endl;
    else
      cout << "Down-sampling:                               " << _downSample << endl;
    if(_convertToGrayscale)
      cout << "Converting query image to grayscale" << endl;
    if(_renderBlurred)                                              //  Are we rendering the blurred source
      cout << "Render:                                    \"blur.png\"" << endl;
    if(_renderFeatures)                                             //  Are we rendering the features on the source
      cout << "Render:                                    \"detected.png\"" << endl;
    if(_renderInliers)                                              //  Are we rendering the inliers on the source
      cout << "Render:                                    \"*.inliers.png\"" << endl;
    if(_writeObjCorr)                                               //  Show me whether we are writing correspondences to file
      cout << "Output:                                    \"*.correspondences\"" << endl;
    if(_writeObjInliers)                                            //  Show me whether we are writing inliers to file
      cout << "Output:                                    \"*.inliers\"" << endl;

    return;
  }

void PnPConfig::paramUsage() const
  {
    cout << "Flags:  -Q             Following argument is the file path for a query image." << endl;
    cout << "        -K             Following argument is the name of the intrinsics file." << endl;
    cout << "                       (By default, program assumes a file named \"K.dat\" is present.)" << endl;
    cout << "        -Rt            Following argument is the name of an extrinsic matrix file." << endl;
    cout << "                       This is used as an initial guess for pose estimation." << endl;
    cout << "        -dist          Following argument is the name of a distortion coefficients file." << endl;
    cout << "                       (By default, no distortion is assumed.)" << endl;
    cout << "        -kd            Following argument is the number of KD-trees to use in our feature index." << endl;
    cout << "                       (Default is 4)." << endl;
    cout << "        -lsh           Following argument is the number of hash tables to use in our LSH feature index." << endl;
    cout << "                       (Default is 10)." << endl;
    cout << "        -pemeth        Following argument is the pose estimation method to use:" << endl;
    cout << "                       {VOTES, INDEP} When set to VOTES (default) poses are estimated using only those correspondences" << endl;
    cout << "                       that voted for each object. When set to INDEP, new correspondences between the query and each" << endl;
    cout << "                       object in isolation are used to estimate pose." << endl;
    cout << "        -iter          Following argument is the number of RANSAC iterations to run. The default is 2000." << endl;
    cout << "        -rErr          Following argument sets the reprojection error. The default is 2 pixels." << endl;
    cout << "        -top           Following argument sets the number of top candidate objects to consider for pose estimation." << endl;
    cout << "                       The default is 5." << endl;
    cout << "        -conf          Following argument sets estimate confidence: stop RANSAC upon finding a model this sure." << endl;
    cout << "                       The default is 0.9999." << endl;
    cout << "        -ratio         Following argument sets the ratio threshold for nearest neighbor feature matches in both" << endl;
    cout << "                       Signature-->Query and in Query-->Signature matching." << endl;
    cout << "        -ratioSig2Q    Following argument sets the ratio threshold for nearest neighbor feature matches for" << endl;
    cout << "                       Signature-->Query only. (Default is 1.0)." << endl;
    cout << "        -ratioQ2Sig    Following argument sets the ratio threshold for nearest neighbor feature matches for" << endl;
    cout << "                       Query-->Signature only. (Default is 0.9)." << endl;
    cout << "        -mmeth         Following string sets the descriptor-matching method." << endl;
    cout << "                       {MUT, Q2SIG, SIG2Q} MUT is default, meaning we make mutual matches." << endl;
    cout << "        -maxNNL2       Following argument sets the maximum distance two L2 (SIFT, SURF) descriptors can be" << endl;
    cout << "                       and still be considered nearest neighbors. (Default is inf, no maximum)." << endl;
    cout << "        -maxNNHamm     Following argument sets the maximum distance two Hamming-distnace (BRISK, ORB) descriptors can be" << endl;
    cout << "                       and still be considered nearest neighbors. (Default is inf, no maximum)." << endl;
    cout << "        -maxQ          Following argument sets the maximum number of features (per descriptor) to extract from the query image." << endl;
    cout << "                       (Default is 0 = no maximum)" << endl;
    cout << "        -nms           Following integer >= 0 is the radius in pixels around which weaker features are suppressed." << endl;
    cout << "                       (Default is 0 = no suppression.)" << endl;
    cout << "        -meth          Following argument is a string that indicates how the PnP solver should work:" << endl;
    cout << "                       {P3P, EPNP, ITERATIVE, DLS, UPNP, AP3P, IPPE, IPPESQ}. Default is AP3P." << endl;
    cout << "        -bkernel       Following argument is the size of the blur filter.";
    cout << "                       The default is 6. A value of 0 = no blur." << endl;
    cout << "        -bmeth         Following argument is a string indicating which blur method to use on the query image." << endl;
    cout << "                       {GAUSS, BOX, MED} BOX is default." << endl;
    cout << "        -gray          Convert query image to grayscale before running feature detection." << endl;
    cout << "        -down          Following float < 1.0 scales down the query image." << endl;
    cout << "                       Down-sampling can save time when applying a blur and can smooth away some distracting details." << endl;
    cout << "                       The default is 0.5. A value of 1.0 = no scaling." << endl;
    cout << "        -refine        Refine the pose estimate by running RANSAC a second time on its preferred inliers." << endl;
    cout << "                       This can lead to improved pose estimates at the cost of a bit of time." << endl;
    cout << "        -BRISKcfg      Following argument is the file path for a (binary) BRISK configuration file." << endl;
    cout << "        -ORBcfg        Following argument is the file path for a (binary) ORB configuration file." << endl;
    cout << "        -SIFTcfg       Following argument is the file path for a (binary) SIFT configuration file." << endl;
    cout << "        -SURFcfg       Following argument is the file path for a (binary) SURF configuration file." << endl;
    cout << "        -mask          Following argument is the file path for a mask image to be applied to the detector." << endl;
    cout << "        -showBlur      Render a new image, \"blur.png\", showing the blurred query image." << endl;
    cout << "        -showFeatures  Render a new image, \"features.png\", showing detected features in the (blurred) query image." << endl;
    cout << "        -showInliers   Render a new image, \"inliers.png\", showing inliers in the (blurred) query image." << endl;
    cout << "        -outFeat       Write 2D interest points detected in query image to file." << endl;
    cout << "        -outCorr       Write each of the top candidates' (pre-RANSAC) 2D-3D correspondences to file." << endl;
    cout << "        -outInlier     Write each of the top candidates' inlying (post-RANSAC) 2D-3D correspondences to file." << endl;
    cout << "        -?" << endl;
    cout << "        -help" << endl;
    cout << "        --help         Displays this message." << endl;
    return;
  }

#endif