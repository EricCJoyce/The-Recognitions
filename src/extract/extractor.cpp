#ifndef __EXTRACTOR_CPP
#define __EXTRACTOR_CPP

#include "extractor.h"

/**************************************************************************************************
 Constructors  */

/* Extractor constructor, no data given */
Extractor::Extractor()
  {
    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::Extractor()" << endl;
    #endif

    unsigned char i;

    if((descriptorCtr = (unsigned int*)malloc(_DESCRIPTOR_TOTAL * sizeof(int))) == NULL)
      {
        cout << "ERROR: Unable to allocate signature descriptor-count vector." << endl;
        exit(1);
      }
    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)                          //  Blank out descriptor counters
      descriptorCtr[i] = 0;

    d = NULL;                                                       //  Nothing detected yet

    useBRISK = true;                                                //  By default, use all detector/descriptors
    useORB = true;
    useSIFT = true;
    useSURF = true;

    limitFeatures = 0;                                              //  Defaults
    blurMethod = _BOX_BLUR;
    blurKernelSize = 0;
    downSample = 1.0;
    renderBlurred = false;
    renderDetections = false;

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

/* Extractor constructor, no data given */
Extractor::Extractor(unsigned char flagArray)
  {
    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::Extractor(" << +flagArray << ")" << endl;
    #endif

    unsigned char i;

    if((descriptorCtr = (unsigned int*)malloc(_DESCRIPTOR_TOTAL * sizeof(int))) == NULL)
      {
        cout << "ERROR: Unable to allocate signature descriptor-count vector." << endl;
        exit(1);
      }
    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)                          //  Blank out descriptor counters
      descriptorCtr[i] = 0;

    d = NULL;                                                       //  Nothing detected yet

    useBRISK = ((flagArray & _BRISK_FLAG) == _BRISK_FLAG);
    useORB   = ((flagArray & _ORB_FLAG)   == _ORB_FLAG);
    useSIFT  = ((flagArray & _SIFT_FLAG)  == _SIFT_FLAG);
    useSURF  = ((flagArray & _SURF_FLAG)  == _SURF_FLAG);

    limitFeatures = 0;                                              //  Defaults
    blurMethod = _BOX_BLUR;
    blurKernelSize = 0;
    downSample = 1.0;
    renderBlurred = false;
    renderDetections = false;

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

/**************************************************************************************************
 Destructor  */

Extractor::~Extractor()
  {
    free(descriptorCtr);
  }

/**************************************************************************************************
 Detection and Extraction  */

void Extractor::initDetectors()
  {
    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::initDetectors()" << endl;
    #endif

    if(useBRISK)                                                    //  Initialize BRISK detector
      brisk_detect = BRISK::create(_brisk_threshold, _brisk_octaves, _brisk_patternScale);

    if(useORB)                                                      //  Initialize ORB detector
      {
        if(limitFeatures == 0)                                      //  Zero does not mean here what it means for SIFT and SURF
          {
            orb_detect = ORB::create(_ORB_DETECTION_DEFAULT_MAXIMUM,
                                     _orb_scaleFactor,
                                     _orb_levels,
                                     _orb_edgeThreshold,
                                     _orb_firstLevel,
                                     _orb_wta_k,
                                     _orb_scoreType,
                                     _orb_patchSize,
                                     _orb_fastThreshold);
          }
        else
          {
            orb_detect = ORB::create(limitFeatures,
                                     _orb_scaleFactor,
                                     _orb_levels,
                                     _orb_edgeThreshold,
                                     _orb_firstLevel,
                                     _orb_wta_k,
                                     _orb_scoreType,
                                     _orb_patchSize,
                                     _orb_fastThreshold);
          }
      }

    if(useSIFT)                                                     //  Initialize SIFT detector
      sift_detect = xfeatures2d::SIFT::create(limitFeatures, _sift_octaveLayers, _sift_contrastThreshold, _sift_edgeThreshold, _sift_sigma);

    if(useSURF)                                                     //  Initialize SURF detector
      surf_detect = xfeatures2d::SURF::create(_surf_hessianThreshold, _surf_octaves, _surf_octaveLayers, false, false);

    return;
  }

/* This image MAY be downsampled. Act according to the internal attribute 'downSample'. */
unsigned int Extractor::extract(cv::Mat img)
  {
    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::extract(width " << +img.cols << ", height " << +img.rows << ")" << endl;
    #endif

    unsigned int totalFeatures;
    unsigned int i;
    unsigned char j;
    cv::Mat outimg;

    cv::Mat briskDesc;                                              //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> briskKP;                              //  Vector of key-points returned by detector of choice on a single image

    cv::Mat orbDesc;                                                //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> orbKP;                                //  Vector of key-points returned by detector of choice on a single image

    cv::Mat siftDesc;                                               //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> siftKP;                               //  Vector of key-points returned by detector of choice on a single image

    cv::Mat surfDesc;                                               //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> surfKP;                               //  Vector of key-points returned by detector of choice on a single image

    unsigned char* ucharBuffer;                                     //  Used to load in BRISK and ORB vectors
    float* floatBuffer;                                             //  Used to load in SIFT and SURF vectors

    reset();                                                        //  Lingering results of a previous detection?

    if(blurKernelSize > 0)
      {
        switch(blurMethod)
          {
            case _BOX_BLUR:
                                  #ifdef __EXTRACTOR_DEBUG
                                  cout << "    Applying (" << +blurKernelSize << " x " << +blurKernelSize << ") box-filter blur." << endl;
                                  #endif

                                  cv::blur(img, img, Size(blurKernelSize, blurKernelSize));
                                  break;
            case _GAUSSIAN_BLUR:
                                  #ifdef __EXTRACTOR_DEBUG
                                  cout << "    Applying (" << +blurKernelSize << " x " << +blurKernelSize << ") Gaussian blur." << endl;
                                  #endif

                                  cv::GaussianBlur(img, img, Size(0, 0), blurKernelSize, blurKernelSize);
                                  break;
            case _MEDIAN_BLUR:
                                  #ifdef __EXTRACTOR_DEBUG
                                  cout << "    Applying (" << +blurKernelSize << " x " << +blurKernelSize << ") median blur." << endl;
                                  #endif

                                  cv::medianBlur(img, img, blurKernelSize);
                                  break;
          }

        if(renderBlurred)
          cv::imwrite("blur.png", img);
      }

    if(useBRISK)                                                    //  Using BRISK? Detect in BRISK.
      {
        brisk_detect->detectAndCompute(img, cv::noArray(), briskKP, briskDesc);
                                                                    //  DO NOT convert to float-32s
        descriptorCtr[_BRISK] = (unsigned int)briskKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << briskKP.size() << " BRISK descriptors from query image." << endl;
        #endif
      }

    if(useORB)                                                      //  Using ORB? Detect in ORB.
      {
        orb_detect->detectAndCompute(img, cv::noArray(), orbKP, orbDesc);
                                                                    //  DO NOT convert to float-32s
        descriptorCtr[_ORB] = (unsigned int)orbKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << orbKP.size() << " ORB descriptors from query image." << endl;
        #endif
      }

    if(useSIFT)                                                     //  Using SIFT? Detect in SIFT.
      {
        sift_detect->detectAndCompute(img, cv::noArray(), siftKP, siftDesc);

        siftDesc.convertTo(siftDesc, CV_32F);                       //  Convert these to float-32s
        descriptorCtr[_SIFT] = (unsigned int)siftKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << siftKP.size() << " SIFT descriptors from query image." << endl;
        #endif
      }

    if(useSURF)                                                     //  Using SURF? Detect in SURF.
      {
        surf_detect->detectAndCompute(img, cv::noArray(), surfKP, surfDesc);
        surfDesc.convertTo(surfDesc, CV_32F);                       //  Convert these to float-32s
        descriptorCtr[_SURF] = (unsigned int)surfKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << surfKP.size() << " SURF descriptors from query image." << endl;
        #endif
      }

    totalFeatures = features();                                     //  Count up total number of features detected

    if(renderDetections)
      {
        outimg = img.clone();
        if(img.channels() == 3)
          cv::cvtColor(outimg, outimg, CV_RGB2GRAY);                //  Convert to grayscale then back to color
        cv::cvtColor(outimg, outimg, CV_GRAY2RGB);                  //  so the keypoints stand out
      }

    if(totalFeatures > 0)                                           //  Were there any features?
      {
                                                                    //  We've counted, now allocate
        if((d = (Descriptor**)malloc(totalFeatures * sizeof(Descriptor*))) == NULL)
          {
            cout << "ERROR: Unable to allocate descriptor-pointer buffer." << endl;
            return 0;
          }

        totalFeatures = 0;                                          //  Reset

        if(useBRISK && briskKP.size() > 0)                          //  Did we use BRISK? Did any turn up?
          {
            if((ucharBuffer = (unsigned char*)malloc(_BRISK_DESCLEN * sizeof(char))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for BRISK descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < briskKP.size(); i++)
              {
                for(j = 0; j < _BRISK_DESCLEN; j++)
                  ucharBuffer[j] = briskDesc.at<unsigned char>(i, j);
                d[totalFeatures] = new BRISKDesc(ucharBuffer);      //  Create a new (BRISK) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(briskKP.at(i).pt.x / downSample, briskKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(briskKP.at(i).size);      //  Save its details
                d[totalFeatures]->setAngle(briskKP.at(i).angle);
                d[totalFeatures]->setResponse(briskKP.at(i).response);
                d[totalFeatures]->setOctave(briskKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(briskKP.at(i).pt.x, briskKP.at(i).pt.y), 3.0,
                                       Scalar(255, 0, 0), 2, 8);    //  BRISK is blue (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(ucharBuffer);
          }

        if(useORB && orbKP.size() > 0)                              //  Did we use ORB? Did any turn up?
          {
            if((ucharBuffer = (unsigned char*)malloc(_ORB_DESCLEN * sizeof(char))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for ORB descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < orbKP.size(); i++)
              {
                for(j = 0; j < _ORB_DESCLEN; j++)
                  ucharBuffer[j] = orbDesc.at<unsigned char>(i, j);
                d[totalFeatures] = new ORBDesc(ucharBuffer);        //  Create a new (ORB) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(orbKP.at(i).pt.x / downSample, orbKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(orbKP.at(i).size);        //  Save its details
                d[totalFeatures]->setAngle(orbKP.at(i).angle);
                d[totalFeatures]->setResponse(orbKP.at(i).response);
                d[totalFeatures]->setOctave(orbKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(orbKP.at(i).pt.x, orbKP.at(i).pt.y), 3.0,
                                       Scalar(0, 255, 0), 2, 8);    //  ORB is green (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(ucharBuffer);
          }

        if(useSIFT && siftKP.size() > 0)                            //  Did we use SIFT? Did any turn up?
          {
            if((floatBuffer = (float*)malloc(_SIFT_DESCLEN * sizeof(float))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for SIFT descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < siftKP.size(); i++)
              {
                for(j = 0; j < _SIFT_DESCLEN; j++)
                  floatBuffer[j] = siftDesc.at<float>(i, j);
                d[totalFeatures] = new SIFTDesc(floatBuffer);       //  Create a new (SIFT) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(siftKP.at(i).pt.x / downSample, siftKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(siftKP.at(i).size);       //  Save its details
                d[totalFeatures]->setAngle(siftKP.at(i).angle);
                d[totalFeatures]->setResponse(siftKP.at(i).response);
                d[totalFeatures]->setOctave(siftKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(siftKP.at(i).pt.x, siftKP.at(i).pt.y), 3.0,
                                       Scalar(0, 0, 255), 2, 8);    //  SIFT is red (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(floatBuffer);
          }

        if(useSURF && surfKP.size() > 0)                            //  Did we use SURF? Did any turn up?
          {
            if((floatBuffer = (float*)malloc(_SURF_DESCLEN * sizeof(float))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for SURF descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < surfKP.size(); i++)
              {
                for(j = 0; j < _SURF_DESCLEN; j++)
                  floatBuffer[j] = surfDesc.at<float>(i, j);
                d[totalFeatures] = new SURFDesc(floatBuffer);       //  Create a new (SURF) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(surfKP.at(i).pt.x / downSample, surfKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(surfKP.at(i).size);       //  Save its details
                d[totalFeatures]->setAngle(surfKP.at(i).angle);
                d[totalFeatures]->setResponse(surfKP.at(i).response);
                d[totalFeatures]->setOctave(surfKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(surfKP.at(i).pt.x, surfKP.at(i).pt.y), 3.0,
                                       Scalar(0, 255, 255), 2, 8);  //  SURF is yellow (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(floatBuffer);
          }
      }

    if(renderDetections)
      cv::imwrite("detected.png", outimg);

    return totalFeatures;
  }

/* This image MAY be downsampled. Act according to the internal attribute 'downSample'.
   The mask must be the same size as the potentially down-sampled image. */
unsigned int Extractor::extract(cv::Mat img, cv::Mat mask)
  {
    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::extract(width " << +img.cols << ", height " << +img.rows << ", with a mask)" << endl;
    #endif

    unsigned int totalFeatures;
    unsigned int i;
    unsigned char j;
    cv::Mat outimg;

    cv::Mat briskDesc;                                              //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> briskKP;                              //  Vector of key-points returned by detector of choice on a single image

    cv::Mat orbDesc;                                                //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> orbKP;                                //  Vector of key-points returned by detector of choice on a single image

    cv::Mat siftDesc;                                               //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> siftKP;                               //  Vector of key-points returned by detector of choice on a single image

    cv::Mat surfDesc;                                               //  To be an N by DESC_LEN matrix for N features in the query image
    std::vector<cv::KeyPoint> surfKP;                               //  Vector of key-points returned by detector of choice on a single image

    unsigned char* ucharBuffer;                                     //  Used to load in BRISK and ORB vectors
    float* floatBuffer;                                             //  Used to load in SIFT and SURF vectors

    reset();                                                        //  Lingering results of a previous detection?

    if(blurKernelSize > 0)
      {
        switch(blurMethod)
          {
            case _BOX_BLUR:
                                  #ifdef __EXTRACTOR_DEBUG
                                  cout << "    Applying (" << +blurKernelSize << " x " << +blurKernelSize << ") box-filter blur." << endl;
                                  #endif

                                  cv::blur(img, img, Size(blurKernelSize, blurKernelSize));
                                  break;
            case _GAUSSIAN_BLUR:
                                  #ifdef __EXTRACTOR_DEBUG
                                  cout << "    Applying (" << +blurKernelSize << " x " << +blurKernelSize << ") Gaussian blur." << endl;
                                  #endif

                                  cv::GaussianBlur(img, img, Size(0, 0), blurKernelSize, blurKernelSize);
                                  break;
            case _MEDIAN_BLUR:
                                  #ifdef __EXTRACTOR_DEBUG
                                  cout << "    Applying (" << +blurKernelSize << " x " << +blurKernelSize << ") median blur." << endl;
                                  #endif

                                  cv::medianBlur(img, img, blurKernelSize);
                                  break;
          }

        if(renderBlurred)
          cv::imwrite("blur.png", img);
      }

    if(useBRISK)                                                    //  Using BRISK? Detect in BRISK.
      {
        brisk_detect->detectAndCompute(img, mask, briskKP, briskDesc);
                                                                    //  DO NOT convert to float-32s
        descriptorCtr[_BRISK] = (unsigned int)briskKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << briskKP.size() << " BRISK descriptors from query image." << endl;
        #endif
      }

    if(useORB)                                                      //  Using ORB? Detect in ORB.
      {
        orb_detect->detectAndCompute(img, mask, orbKP, orbDesc);
                                                                    //  DO NOT convert to float-32s
        descriptorCtr[_ORB] = (unsigned int)orbKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << orbKP.size() << " ORB descriptors from query image." << endl;
        #endif
      }

    if(useSIFT)                                                     //  Using SIFT? Detect in SIFT.
      {
        sift_detect->detectAndCompute(img, mask, siftKP, siftDesc);

        siftDesc.convertTo(siftDesc, CV_32F);                       //  Convert these to float-32s
        descriptorCtr[_SIFT] = (unsigned int)siftKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << siftKP.size() << " SIFT descriptors from query image." << endl;
        #endif
      }

    if(useSURF)                                                     //  Using SURF? Detect in SURF.
      {
        surf_detect->detectAndCompute(img, mask, surfKP, surfDesc);
        surfDesc.convertTo(surfDesc, CV_32F);                       //  Convert these to float-32s
        descriptorCtr[_SURF] = (unsigned int)surfKP.size();

        #ifdef __EXTRACTOR_DEBUG
        cout << "  Collected " << surfKP.size() << " SURF descriptors from query image." << endl;
        #endif
      }

    totalFeatures = features();                                     //  Count up total number of features detected

    if(renderDetections)
      {
        outimg = img.clone();
        if(img.channels() == 3)
          cv::cvtColor(outimg, outimg, CV_RGB2GRAY);                //  Convert to grayscale then back to color
        cv::cvtColor(outimg, outimg, CV_GRAY2RGB);                  //  so the keypoints stand out
      }

    if(totalFeatures > 0)                                           //  Were there any features?
      {
                                                                    //  We've counted, now allocate
        if((d = (Descriptor**)malloc(totalFeatures * sizeof(Descriptor*))) == NULL)
          {
            cout << "ERROR: Unable to allocate descriptor-pointer buffer." << endl;
            return 0;
          }

        totalFeatures = 0;                                          //  Reset

        if(useBRISK && briskKP.size() > 0)                          //  Did we use BRISK? Did any turn up?
          {
            if((ucharBuffer = (unsigned char*)malloc(_BRISK_DESCLEN * sizeof(char))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for BRISK descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < briskKP.size(); i++)
              {
                for(j = 0; j < _BRISK_DESCLEN; j++)
                  ucharBuffer[j] = briskDesc.at<unsigned char>(i, j);
                d[totalFeatures] = new BRISKDesc(ucharBuffer);      //  Create a new (BRISK) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(briskKP.at(i).pt.x / downSample, briskKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(briskKP.at(i).size);      //  Save its details
                d[totalFeatures]->setAngle(briskKP.at(i).angle);
                d[totalFeatures]->setResponse(briskKP.at(i).response);
                d[totalFeatures]->setOctave(briskKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(briskKP.at(i).pt.x, briskKP.at(i).pt.y), 3.0,
                                       Scalar(255, 0, 0), 2, 8);    //  BRISK is blue (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(ucharBuffer);
          }

        if(useORB && orbKP.size() > 0)                              //  Did we use ORB? Did any turn up?
          {
            if((ucharBuffer = (unsigned char*)malloc(_ORB_DESCLEN * sizeof(char))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for ORB descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < orbKP.size(); i++)
              {
                for(j = 0; j < _ORB_DESCLEN; j++)
                  ucharBuffer[j] = orbDesc.at<unsigned char>(i, j);
                d[totalFeatures] = new ORBDesc(ucharBuffer);        //  Create a new (ORB) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(orbKP.at(i).pt.x / downSample, orbKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(orbKP.at(i).size);        //  Save its details
                d[totalFeatures]->setAngle(orbKP.at(i).angle);
                d[totalFeatures]->setResponse(orbKP.at(i).response);
                d[totalFeatures]->setOctave(orbKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(orbKP.at(i).pt.x, orbKP.at(i).pt.y), 3.0,
                                       Scalar(0, 255, 0), 2, 8);    //  ORB is green (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(ucharBuffer);
          }

        if(useSIFT && siftKP.size() > 0)                            //  Did we use SIFT? Did any turn up?
          {
            if((floatBuffer = (float*)malloc(_SIFT_DESCLEN * sizeof(float))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for SIFT descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < siftKP.size(); i++)
              {
                for(j = 0; j < _SIFT_DESCLEN; j++)
                  floatBuffer[j] = siftDesc.at<float>(i, j);
                d[totalFeatures] = new SIFTDesc(floatBuffer);       //  Create a new (SIFT) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(siftKP.at(i).pt.x / downSample, siftKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(siftKP.at(i).size);       //  Save its details
                d[totalFeatures]->setAngle(siftKP.at(i).angle);
                d[totalFeatures]->setResponse(siftKP.at(i).response);
                d[totalFeatures]->setOctave(siftKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(siftKP.at(i).pt.x, siftKP.at(i).pt.y), 3.0,
                                       Scalar(0, 0, 255), 2, 8);    //  SIFT is red (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(floatBuffer);
          }

        if(useSURF && surfKP.size() > 0)                            //  Did we use SURF? Did any turn up?
          {
            if((floatBuffer = (float*)malloc(_SURF_DESCLEN * sizeof(float))) == NULL)
              {
                cout << "ERROR: Unable to allocate temporary buffer for SURF descriptor-vectors." << endl;
                return 0;
              }

            for(i = 0; i < surfKP.size(); i++)
              {
                for(j = 0; j < _SURF_DESCLEN; j++)
                  floatBuffer[j] = surfDesc.at<float>(i, j);
                d[totalFeatures] = new SURFDesc(floatBuffer);       //  Create a new (SURF) descriptor
                                                                    //  Save its (TRUE, REGARDLESS OF DOWN-SAMPLING) 2D location
                d[totalFeatures]->XYZ(surfKP.at(i).pt.x / downSample, surfKP.at(i).pt.y / downSample, INFINITY);
                d[totalFeatures]->setSize(surfKP.at(i).size);       //  Save its details
                d[totalFeatures]->setAngle(surfKP.at(i).angle);
                d[totalFeatures]->setResponse(surfKP.at(i).response);
                d[totalFeatures]->setOctave(surfKP.at(i).octave);

                if(renderDetections)
                  {
                    cv::circle(outimg, Point(surfKP.at(i).pt.x, surfKP.at(i).pt.y), 3.0,
                                       Scalar(0, 255, 255), 2, 8);  //  SURF is yellow (circle uses BGR)
                  }

                totalFeatures++;
              }

            free(floatBuffer);
          }
      }

    if(renderDetections)
      cv::imwrite("detected.png", outimg);

    return totalFeatures;
  }

/* Compare to Signature::toMat().

   Given 'type' in {BRISK, ORB, SIFT, SURF} write an N x D cv::Mat
   where N is the number of features in stored using descriptor 'type'
   and D is the length of that descriptor vector.
   That is, if we are using BRISK and SURF, then descMat(_BRISK, mat) will fill 'mat' with N BRISK row-vectors,
   and descMat(_SURF, mat) will fill 'mat' with N SURF row-vectors. */
void Extractor::descMat(unsigned char type, cv::Mat* mat) const
  {
    unsigned int i, j;
    unsigned char k;
    unsigned int total;

    unsigned char* ucharBuffer;                                     //  Accumulators
    float* floatBuffer;

    unsigned char* ucharTmp;                                        //  Copy the vector from the Descriptor object
    float* floatTmp;

    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::descMat(" << +type << ") to build (" << +descriptorCtr[type] << " x ";
    switch(type)
      {
        case _BRISK:  cout << +_BRISK_DESCLEN;  break;
        case _ORB:    cout << +_ORB_DESCLEN;    break;
        case _SIFT:   cout << +_SIFT_DESCLEN;   break;
        case _SURF:   cout << +_SURF_DESCLEN;   break;
      }
    cout << ") matrix" << endl;
    #endif

    if(descriptorCtr[type] > 0)                                     //  Are there any descriptors of the given type?
      {
        if(type == _BRISK)                                          //  Allocate the right buffer, the right amount
          {
            if((ucharBuffer = (unsigned char*)malloc(descriptorCtr[_BRISK] * _BRISK_DESCLEN * sizeof(char))) == NULL)
              {
                cout << "ERROR: Unable to allocate unsigned char buffer for BRISK output to OpenCV Mat." << endl;
                return ;
              }
          }
        else if(type == _ORB)
          {
            if((ucharBuffer = (unsigned char*)malloc(descriptorCtr[_ORB] * _ORB_DESCLEN * sizeof(char))) == NULL)
              {
                cout << "ERROR: Unable to allocate unsigned char buffer for ORB output to OpenCV Mat." << endl;
                return ;
              }
          }
        else if(type == _SIFT)
          {
            if((floatBuffer = (float*)malloc(descriptorCtr[_SIFT] * _SIFT_DESCLEN * sizeof(float))) == NULL)
              {
                cout << "ERROR: Unable to allocate unsigned char buffer for SIFT output to OpenCV Mat." << endl;
                return ;
              }
          }
        else if(type == _SURF)
          {
            if((floatBuffer = (float*)malloc(descriptorCtr[_SURF] * _SURF_DESCLEN * sizeof(float))) == NULL)
              {
                cout << "ERROR: Unable to allocate unsigned char buffer for SURF output to OpenCV Mat." << endl;
                return ;
              }
          }

        total = features();
        j = 0;                                                      //  Offset into the buffer
        for(i = 0; i < total; i++)                                  //  Now that we've allocated space, fill in that space
          {
            if(d[i]->type() == type)                                //  Type matched
              {
                if(type == _BRISK)                                  //  That type is BRISK
                  {
                    d[i]->vec(&ucharTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _BRISK_DESCLEN; k++)             //  Copy the temp into the total buffer
                      ucharBuffer[j * _BRISK_DESCLEN + k] = ucharTmp[k];
                    free(ucharTmp);                                 //  Dump temp
                  }
                else if(type == _ORB)                               //  That type is ORB
                  {
                    d[i]->vec(&ucharTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _ORB_DESCLEN; k++)               //  Copy the temp into the total buffer
                      ucharBuffer[j * _ORB_DESCLEN + k] = ucharTmp[k];
                    free(ucharTmp);                                 //  Dump temp
                  }
                else if(type == _SIFT)                              //  That type is SIFT
                  {
                    d[i]->vec(&floatTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _SIFT_DESCLEN; k++)              //  Copy the temp into the total buffer
                      floatBuffer[j * _SIFT_DESCLEN + k] = floatTmp[k];
                    free(floatTmp);                                 //  Dump temp
                  }
                else if(type == _SURF)                              //  That type is SURF
                  {
                    d[i]->vec(&floatTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _SURF_DESCLEN; k++)              //  Copy the temp into the total buffer
                      floatBuffer[j * _SURF_DESCLEN + k] = floatTmp[k];
                    free(floatTmp);                                 //  Dump temp
                  }

                j++;                                                //  Increment offset into buffer
              }
          }

        if(type == _BRISK)
          {
            (*mat) = cv::Mat(descriptorCtr[_BRISK], _BRISK_DESCLEN, CV_8U);
            memcpy(mat->data, ucharBuffer, descriptorCtr[_BRISK] * _BRISK_DESCLEN * sizeof(char));
            free(ucharBuffer);
          }
        else if(type == _ORB)
          {
            (*mat) = cv::Mat(descriptorCtr[_ORB], _ORB_DESCLEN, CV_8U);
            memcpy(mat->data, ucharBuffer, descriptorCtr[_ORB] * _ORB_DESCLEN * sizeof(char));
            free(ucharBuffer);
          }
        else if(type == _SIFT)
          {
            (*mat) = cv::Mat(descriptorCtr[_SIFT], _SIFT_DESCLEN, CV_32F);
            memcpy(mat->data, floatBuffer, descriptorCtr[_SIFT] * _SIFT_DESCLEN * sizeof(float));
            free(floatBuffer);
          }
        else if(type == _SURF)
          {
            (*mat) = cv::Mat(descriptorCtr[_SURF], _SURF_DESCLEN, CV_32F);
            memcpy(mat->data, floatBuffer, descriptorCtr[_SURF] * _SURF_DESCLEN * sizeof(float));
            free(floatBuffer);
          }
      }

    #ifdef __EXTRACTOR_DEBUG
    cout << "  Extractor::descMat(" << +type << ") built (" << mat->rows << " x " << mat->cols << ") matrix" << endl;
    #endif

    return;
  }

/* Add up counts of all Descriptor types extracted from query image */
unsigned int Extractor::features() const
  {
    unsigned int i = 0;
    unsigned char j;

    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::features(): ";
    #endif

    for(j = 0; j < _DESCRIPTOR_TOTAL; j++)
      i += descriptorCtr[j];

    #ifdef __EXTRACTOR_DEBUG
    cout << +i << endl;
    #endif

    return i;
  }

/* Add up counts of all Descriptors of given type extracted from query image */
unsigned int Extractor::features(unsigned char type) const
  {
    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::features(" << +type << "): ";
    #endif

    return descriptorCtr[type];
  }

/**************************************************************************************************
 Setters  */

void Extractor::setFlags(unsigned char flagArray)
  {
    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::setFlags(" << +flagArray << ")" << endl;
    #endif

    useBRISK = ((flagArray & _BRISK_FLAG) == _BRISK_FLAG);
    useORB   = ((flagArray & _ORB_FLAG)   == _ORB_FLAG);
    useSIFT  = ((flagArray & _SIFT_FLAG)  == _SIFT_FLAG);
    useSURF  = ((flagArray & _SURF_FLAG)  == _SURF_FLAG);

    return;
  }

void Extractor::setBRISK(bool b)
  {
    useBRISK = b;
    return;
  }

void Extractor::setORB(bool b)
  {
    useORB = b;
    return;
  }

void Extractor::setSIFT(bool b)
  {
    useSIFT = b;
    return;
  }

void Extractor::setSURF(bool b)
  {
    useSURF = b;
    return;
  }

void Extractor::setFeatureLimit(unsigned int q)
  {
    limitFeatures = q;
    return;
  }

void Extractor::setBRISKparams(PnPConfig* config)
  {
    _brisk_threshold = config->brisk_threshold();
    _brisk_octaves = config->brisk_octaves();
    _brisk_patternScale = config->brisk_patternScale();
    return;
  }

void Extractor::setORBparams(PnPConfig* config)
  {
    _orb_scaleFactor = config->orb_scaleFactor();
    _orb_levels = config->orb_levels();
    _orb_edgeThreshold = config->orb_edgeThreshold();
    _orb_firstLevel = config->orb_firstLevel();
    _orb_wta_k = config->orb_wta_k();
    _orb_scoreType = config->orb_scoreType();
    _orb_patchSize = config->orb_patchSize();
    _orb_fastThreshold = config->orb_fastThreshold();
    return;
  }

void Extractor::setSIFTparams(PnPConfig* config)
  {
    _sift_octaveLayers = config->sift_octaveLayers();
    _sift_contrastThreshold = config->sift_contrastThreshold();
    _sift_edgeThreshold = config->sift_edgeThreshold();
    _sift_sigma = config->sift_sigma();
    return;
  }

void Extractor::setSURFparams(PnPConfig* config)
  {
    _surf_hessianThreshold = config->surf_hessianThreshold();
    _surf_octaves = config->surf_octaves();
    _surf_octaveLayers = config->surf_octaveLayers();
    return;
  }

void Extractor::disableBlur()
  {
    blurKernelSize = 0;
    return;
  }

void Extractor::setBlurMethod(unsigned char flag)
  {
    if(flag >= _BOX_BLUR && flag <= _MEDIAN_BLUR)
      blurMethod = flag;
    return;
  }

void Extractor::setBlurKernelSize(unsigned char size)
  {
    blurKernelSize = size;
    return;
  }

void Extractor::disableDownSample()
  {
    downSample = 1.0;
    return;
  }

void Extractor::setDownSample(float d)
  {
    if(d <= 1.0 && d > 0.0)
      downSample = d;
    return;
  }

void Extractor::setRenderBlur(bool b)
  {
    renderBlurred = b;
    return;
  }

void Extractor::setRenderDetections(bool b)
  {
    renderDetections = b;
    return;
  }

/**************************************************************************************************
 Getters  */

unsigned int Extractor::featureLimit() const
  {
    return limitFeatures;
  }

bool Extractor::BRISK() const
  {
    return useBRISK;
  }

bool Extractor::ORB() const
  {
    return useORB;
  }

bool Extractor::SIFT() const
  {
    return useSIFT;
  }

bool Extractor::SURF() const
  {
    return useSURF;
  }

unsigned char Extractor::type(unsigned int i) const
  {
    unsigned int lim = features();
    if(i < lim)
      return d[i]->type();
    return _DESCRIPTOR_TOTAL;
  }

float Extractor::x(unsigned int i) const
  {
    unsigned int lim = features();
    if(i < lim)
      return d[i]->x();
    return INFINITY;
  }

float Extractor::y(unsigned int i) const
  {
    unsigned int lim = features();
    if(i < lim)
      return d[i]->y();
    return INFINITY;
  }

float Extractor::size(unsigned int i) const
  {
    unsigned int lim = features();
    if(i < lim)
      return d[i]->size();
    return INFINITY;
  }

float Extractor::angle(unsigned int i) const
  {
    unsigned int lim = features();
    if(i < lim)
      return d[i]->angle();
    return INFINITY;
  }

float Extractor::response(unsigned int i) const
  {
    unsigned int lim = features();
    if(i < lim)
      return d[i]->response();
    return INFINITY;
  }

signed int Extractor::octave(unsigned int i) const
  {
    unsigned int lim = features();
    if(i < lim)
      return d[i]->octave();
    return numeric_limits<unsigned int>::max();
  }

unsigned char Extractor::vec(unsigned int i, void** buffer) const
  {
    unsigned int n;
    unsigned int lim = features();

    if(i < lim)
      {
        switch(d[i]->type())
          {
            case _BRISK: if(((*buffer) = malloc(d[i]->len() * sizeof(char))) == NULL)
                           {
                             #ifdef __EXTRACTOR_DEBUG
                             cout << "ERROR: Unable to allocate unsigned char buffer for BRISK output to void*." << endl;
                             #endif
                             return 0;
                           }
                         for(n = 0; n < d[i]->len(); n++)
                           (*((unsigned char**)buffer))[n] = d[i]->atu(n);
                         break;

            case _ORB:   if(((*buffer) = malloc(d[i]->len() * sizeof(char))) == NULL)
                           {
                             #ifdef __EXTRACTOR_DEBUG
                             cout << "ERROR: Unable to allocate unsigned char buffer for ORB output to void*." << endl;
                             #endif
                             return 0;
                           }
                         for(n = 0; n < d[i]->len(); n++)
                           (*((unsigned char**)buffer))[n] = d[i]->atu(n);
                         break;

            case _SIFT:  if(((*buffer) = malloc(d[i]->len() * sizeof(float))) == NULL)
                           {
                             #ifdef __EXTRACTOR_DEBUG
                             cout << "ERROR: Unable to allocate float buffer for SIFT output to void*." << endl;
                             #endif
                             return 0;
                           }
                         for(n = 0; n < d[i]->len(); n++)
                           (*((float**)buffer))[n] = d[i]->atf(n);
                         break;

            case _SURF:  if(((*buffer) = malloc(d[i]->len() * sizeof(float))) == NULL)
                           {
                             #ifdef __EXTRACTOR_DEBUG
                             cout << "ERROR: Unable to allocate float buffer for SURF output to void*." << endl;
                             #endif
                             return 0;
                           }
                         for(n = 0; n < d[i]->len(); n++)
                           (*((float**)buffer))[n] = d[i]->atf(n);
                         break;
          }

        return d[i]->len();
      }

    return 0;
  }

/**************************************************************************************************
 Utilities  */

void Extractor::reset()
  {
    unsigned int i;
    unsigned int len;

    #ifdef __EXTRACTOR_DEBUG
    cout << "Extractor::reset()" << endl;
    #endif

    if(d != NULL)                                                   //  Do not attempt to empty out a NULL
      {
        len = features();                                           //  Get number of Descriptor-pointers
        for(i = 0; i < len; i++)                                    //  Delete each one
          delete d[i];

        free(d);                                                    //  Free the array

        for(i = 0; i < _DESCRIPTOR_TOTAL; i++)                      //  Reset counters
          descriptorCtr[i] = 0;

        d = NULL;                                                   //  Bury the ashes
      }

    return;
  }

#endif