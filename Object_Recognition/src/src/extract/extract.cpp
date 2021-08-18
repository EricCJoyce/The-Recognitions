/*  Eric C. Joyce, Stevens Institute of Technology, 2020.

    Given an image, identify all SIFT, BRISK, ORB, and SURF features.
    Write interest points' 2D coordinates, detection response details, and feature-description vector:
      2D coordinates:   Detection   Detection  Detection  Detection  Descriptor    Descriptor          .....
                         details:    details:   details:   details:    length       elements
e.g.      x   y           size        angle     response   octave      128         SIFT[0]   SIFT[1]   ...   SIFT[n-1]
       [ floats ]       [float]      [float]   [float]     [int]      [uchar]       [float]   [float]   ...   [float]

                                                                                                    or

                                                                                   ORB[0]    ORB[1]    ...   ORB[n-1]
                                                                                    [uchar]   [uchar]   ...   [uchar]
*/

/*
./extract -Q transformed/control_panel/transformed.270.png -BRISK N -ORB N -SIFT Y -SURF N -mask masks/control_panel/mask.270.png -e 5 -SIFTcfg ../detector_config/sift.dat -D depth/control_panel/depth.270.dat
*/

#include <algorithm>                                                //  Needed for reverse()
#include <fstream>
#include <iomanip>                                                  //  Needed for std::setprecision
#include <iostream>
#include <limits>                                                   //  Needed for int-infinity
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>                                                 //  Do I need this?

#include "descriptor.h"
#include "extractor.h"
#include "pnp_config.h"

#define BLENDER_INFINITY  65504.0                                   /* Number used by Blender for "infinitly far away */

#define PARAM_NONE         0                                        /* Flag indicating that no argument is to follow */
#define PARAM_SIFT         1                                        /* Flag indicating that Y or N for SIFT is to follow */
#define PARAM_ORB          2                                        /* Flag indicating that Y or N for ORB is to follow */
#define PARAM_BRISK        3                                        /* Flag indicating that Y or N for BRISK is to follow */
#define PARAM_SURF         4                                        /* Flag indicating that Y or N for SURF is to follow */
#define PARAM_E            5                                        /* Flag indicating that the erosion value is to follow */
#define PARAM_DEPTH        6                                        /* Flag indicating that the depth buffer is to follow */

/*
#define __EXTRACT_DEBUG 1
*/

using namespace cv;
using namespace std;

/**************************************************************************************************
 Typedefs  */

/**************************************************************************************************
 Prototypes  */

unsigned int loadDepthBuffer(char*, float**);
void parseRunParameters(int, char**, bool**, unsigned char*, char**, unsigned int*, bool*);
void usage(void);

/**************************************************************************************************
 Functions  */

/* Main loads an image and converts it to a byte-array.
   Main ships all these byte-arrays to pnp_main(), which computes a pose and writes to an output buffer. */
int main(int argc, char** argv)
  {
    Extractor* extractor;
    cv::Mat img;
    cv::Mat mask;
    cv::Mat depth;

    PnPConfig config(argc, argv);                                   //  Load parameters into a config object
    bool* descriptors;                                              //  Array of feature descriptors to use, length is _DESCRIPTOR_TOTAL
    unsigned char flags = 0;
    unsigned char erosion = 0;                                      //  The number of pixels by which to erode the image
    bool showMask = false;
    unsigned int Qpathlen = 0;
    char* imgFilename;
    char* maskFilename;
    char* depthFilename;
    unsigned int depthFilenameLen = 0;
    float* depthBuffer;

    float tmpFloat;
    signed int tmpSignedInt;
    unsigned char tmpUchar;
    unsigned char* ucharBuffer;
    float* floatBuffer;
    unsigned int x, y;

    ofstream siftfp;                                                //  If we are using SIFT, then open and write to siftfp.
    ofstream briskfp;                                               //  If we are using BRISK, then open and write to briskfp.
    ofstream orbfp;                                                 //  If we are using ORB, then open and write to orbfp.
    ofstream surffp;                                                //  If we are using SURF, then open and write to surffp.

    unsigned int i;
    unsigned char j;

    if(config.helpme())                                             //  Just displaying help? Do it and quit.
      {
        usage();
        return 0;
      }
                                                                    //  Parse parameters
    parseRunParameters(argc, argv, &descriptors, &erosion, &depthFilename, &depthFilenameLen, &showMask);

    if(depthFilenameLen == 0)
      {
        cout << "ERROR: no depth buffer given" << endl;
        return 0;
      }

    Qpathlen = config.Q(&imgFilename);
    if(Qpathlen == 0)
      {
        cout << "ERROR: no image given" << endl;
        return 0;
      }
    img = cv::imread(imgFilename, CV_LOAD_IMAGE_COLOR);             //  Load image

    if(config.hasMask())
      {
        config.mask(&maskFilename);
        mask = cv::imread(maskFilename, IMREAD_GRAYSCALE);          //  Load mask
      }
    else
      {
        mask = cv::Mat(img.rows, img.cols, CV_8U);
        for(y = 0; y < (unsigned int)img.rows; y++)
          {
            for(x = 0; x < (unsigned int)img.cols; x++)
              mask.at<unsigned char>(y, x) = 255;
          }
      }

    if(loadDepthBuffer(depthFilename, &depthBuffer) == 0)           //  Read and reverse
      {
        cout << "ERROR: Unable to load depth buffer." << endl;
        return 0;
      }

    depth = cv::Mat(img.rows, img.cols, CV_32F, depthBuffer);       //  Pack buffer into matrix
    cv::flip(depth, depth, 1);                                      //  Flip horizontally

    for(y = 0; y < (unsigned int)img.rows; y++)
      {
        for(x = 0; x < (unsigned int)img.cols; x++)
          {
            if(depth.at<float>(y, x) == BLENDER_INFINITY)
              mask.at<unsigned char>(y, x) = 0;
          }
      }

    if(erosion > 0)
      cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, Size(erosion, erosion)));

    if(showMask)
      cv::imwrite("final_mask.png", mask);

    if(descriptors[_BRISK])                                         //  Mark flags
      flags |= 1;
    if(descriptors[_ORB])
      flags |= 2;
    if(descriptors[_SIFT])
      flags |= 4;
    if(descriptors[_SURF])
      flags |= 8;

    if(config.verbose())                                            //  Break it down for 'em
      {
        cout << ">>> Extracting feature descriptors and 2D feature points from the image \"" << imgFilename << "\"." << endl;
        cout << ">>> We will use the depth buffer \"" << depthFilename << "\"" << endl;
        if(config.hasMask())
          cout << ">>> We will apply the mask image \"" << maskFilename << "\"." << endl;
        if(descriptors[_SIFT])
          cout << ">>> We will use SIFT detectors." << endl;
        if(descriptors[_ORB])
          cout << ">>> We will use ORB detectors." << endl;
        if(descriptors[_BRISK])
          cout << ">>> We will use BRISK detectors." << endl;
        if(descriptors[_SURF])
          cout << ">>> We will use SURF detectors." << endl;
        if(config.renderFeatures())
          cout << ">>> We will write detected features to a new image, \"features.png\"." << endl;

        cout << ">>> Starting detection." << endl << endl;
      }

    extractor = new Extractor(flags);                               //  Create an Extractor
    extractor->setBlurMethod(config.blurMethod());                  //  Use the blur method from the Config object
    extractor->setBlurKernelSize(config.blurKernelSize());          //  Use the blur kernel size from the Config object
    extractor->setRenderBlur(config.renderBlurred());               //  Render the blurred query image, according to the Config object
    extractor->setRenderDetections(config.renderFeatures());        //  Render the detected featurs,  according to the Config object
    extractor->setFeatureLimit(config.qLimit());                    //  Clamp the number of features (per descriptor) according to Config obj
    extractor->setBRISKparams(&config);                             //  Configure detectors the way we want
    extractor->setORBparams(&config);
    extractor->setSIFTparams(&config);
    extractor->setSURFparams(&config);
    extractor->initDetectors();                                     //  Initialize detectors
    extractor->extract(img, mask);                                  //  Extract features using a mask

    if(config.verbose())
      {
        cout << endl << "    TOTAL = " << +extractor->features() << " features." << endl;
        cout << endl << ">>> Writing 2D features to file(s)..." << endl;
      }

    if(descriptors[_BRISK])
      briskfp.open("features.brisk");
    if(descriptors[_ORB])
      orbfp.open("features.orb");
    if(descriptors[_SIFT])
      siftfp.open("features.sift");
    if(descriptors[_SURF])
      surffp.open("features.surf");

    for(i = 0; i < extractor->features(); i++)
      {
        switch(extractor->type(i))
          {
            case _BRISK: tmpFloat = extractor->x(i);
                         briskfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->y(i);
                         briskfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->size(i);
                         briskfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->angle(i);
                         briskfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->response(i);
                         briskfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpSignedInt = extractor->octave(i);
                         briskfp.write((char*)(&tmpSignedInt), sizeof(int));
                         tmpUchar = _BRISK_DESCLEN;
                         briskfp.write((char*)(&tmpUchar), sizeof(char));

                         extractor->vec(i, (void**)(&ucharBuffer));
                         for(j = 0; j < _BRISK_DESCLEN; j++)
                           {
                             tmpUchar = ucharBuffer[j];
                             briskfp.write((char*)(&tmpUchar), sizeof(char));
                           }
                         break;

            case _ORB:   tmpFloat = extractor->x(i);
                         orbfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->y(i);
                         orbfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->size(i);
                         orbfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->angle(i);
                         orbfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->response(i);
                         orbfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpSignedInt = extractor->octave(i);
                         orbfp.write((char*)(&tmpSignedInt), sizeof(int));
                         tmpUchar = _ORB_DESCLEN;
                         orbfp.write((char*)(&tmpUchar), sizeof(char));

                         extractor->vec(i, (void**)(&ucharBuffer));
                         for(j = 0; j < _ORB_DESCLEN; j++)
                           {
                             tmpUchar = ucharBuffer[j];
                             orbfp.write((char*)(&tmpUchar), sizeof(char));
                           }
                         break;

            case _SIFT:  tmpFloat = extractor->x(i);
                         siftfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->y(i);
                         siftfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->size(i);
                         siftfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->angle(i);
                         siftfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->response(i);
                         siftfp.write((char*)(&tmpFloat), sizeof(float));
                         tmpSignedInt = extractor->octave(i);
                         siftfp.write((char*)(&tmpSignedInt), sizeof(int));
                         tmpUchar = _SIFT_DESCLEN;
                         siftfp.write((char*)(&tmpUchar), sizeof(char));

                         extractor->vec(i, (void**)(&floatBuffer));
                         for(j = 0; j < _SIFT_DESCLEN; j++)
                           {
                             tmpFloat = floatBuffer[j];
                             siftfp.write((char*)(&tmpFloat), sizeof(float));
                           }
                         break;

            case _SURF:  tmpFloat = extractor->x(i);
                         surffp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->y(i);
                         surffp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->size(i);
                         surffp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->angle(i);
                         surffp.write((char*)(&tmpFloat), sizeof(float));
                         tmpFloat = extractor->response(i);
                         surffp.write((char*)(&tmpFloat), sizeof(float));
                         tmpSignedInt = extractor->octave(i);
                         surffp.write((char*)(&tmpSignedInt), sizeof(int));
                         tmpUchar = _SURF_DESCLEN;
                         surffp.write((char*)(&tmpUchar), sizeof(char));

                         extractor->vec(i, (void**)(&floatBuffer));
                         for(j = 0; j < _SURF_DESCLEN; j++)
                           {
                             tmpFloat = floatBuffer[j];
                             surffp.write((char*)(&tmpFloat), sizeof(float));
                           }
                         break;
          }
      }

    if(descriptors[_BRISK])
      briskfp.close();
    if(descriptors[_ORB])
      orbfp.close();
    if(descriptors[_SIFT])
      siftfp.close();
    if(descriptors[_SURF])
      surffp.close();

    if(_DESCRIPTOR_TOTAL > 0)
      free(descriptors);
    free(imgFilename);
    if(config.hasMask())
      free(maskFilename);

    return 0;
  }

/* The depth buffer is packed little-endian, but must be reversed, reshaped, then flipped horizontally.
   This function handles the reading and reversing. */
unsigned int loadDepthBuffer(char* filename, float** buffer)
  {
    unsigned int len = 0;                                           //  Number of floats
    unsigned int filelen;                                           //  Number of bytes
    unsigned int i;
    float tmp;
    ifstream fh;

    fh.open(filename, ifstream::binary);
    if(fh)
      {
        fh.seekg(0, fh.end);
        filelen = fh.tellg();
        len = filelen / sizeof(float);
        fh.seekg(0, fh.beg);

        if(((*buffer) = (float*)malloc(len * sizeof(float))) == NULL)
          {
            cout << "ERROR: Unable to allocate float array." << endl;
            return 0;
          }
        for(i = 0; i < len; i++)
          {
            fh.read((char*)(&tmp), sizeof(float));
            (*buffer)[i] = tmp;
          }

        reverse((*buffer), (*buffer) + (len - 1));
      }

    return len;
  }

/* Set parameters according to command-line flags and values. */
void parseRunParameters(int argc, char** argv, bool** descriptors, unsigned char* erosion, char** depthFilename, unsigned int* depthFilenameLen, bool* showMask)
  {
    unsigned int i = 1;
    unsigned char argtarget = PARAM_NONE;

    if(((*descriptors) = (bool*)malloc(_DESCRIPTOR_TOTAL * sizeof(bool))) == NULL)
      {
        cout << "ERROR: Unable to allocate feature parameter array." << endl;
        exit(1);
      }
    (*descriptors)[_BRISK] = true;                                  //  By default, use all descriptors
    (*descriptors)[_ORB] = true;
    (*descriptors)[_SIFT] = true;
    (*descriptors)[_SURF] = true;

    while(i < (unsigned int)argc)
      {
        if(strcmp(argv[i], "-SIFT") == 0)                           //  String to follow toggles the use of SIFT
          argtarget = PARAM_SIFT;
        else if(strcmp(argv[i], "-ORB") == 0)                       //  String to follow toggles the use of ORB
          argtarget = PARAM_ORB;
        else if(strcmp(argv[i], "-BRISK") == 0)                     //  String to follow toggles the use of BRISK
          argtarget = PARAM_BRISK;
        else if(strcmp(argv[i], "-SURF") == 0)                      //  String to follow toggles the use of SURF
          argtarget = PARAM_SURF;
        else if(strcmp(argv[i], "-D") == 0)                         //  String to follow is the file path to the depth buffer
          argtarget = PARAM_DEPTH;
        else if(strcmp(argv[i], "-e") == 0)                         //  Following integer is the number of pixels to erode
          argtarget = PARAM_E;
        else if(strcmp(argv[i], "-showMask") == 0)                  //  Write the mask to image
          (*showMask) = true;
        else                                                        //  Not one of our flags... react to one of the flags
          {
            switch(argtarget)
              {
                case PARAM_SIFT:                                    //  Incoming {"Y", "y", "N", "n"} for SIFT
                  if(argv[i][0] == 'Y' || argv[i][0] == 'y')
                    (*descriptors)[_SIFT] = true;
                  else if(argv[i][0] == 'N' || argv[i][0] == 'n')
                    (*descriptors)[_SIFT] = false;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_ORB:                                     //  Incoming {"Y", "y", "N", "n"} for ORB
                  if(argv[i][0] == 'Y' || argv[i][0] == 'y')
                    (*descriptors)[_ORB] = true;
                  else if(argv[i][0] == 'N' || argv[i][0] == 'n')
                    (*descriptors)[_ORB] = false;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_BRISK:                                   //  Incoming {"Y", "y", "N", "n"} for BRISK
                  if(argv[i][0] == 'Y' || argv[i][0] == 'y')
                    (*descriptors)[_BRISK] = true;
                  else if(argv[i][0] == 'N' || argv[i][0] == 'n')
                    (*descriptors)[_BRISK] = false;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_SURF:                                    //  Incoming {"Y", "y", "N", "n"} for SURF
                  if(argv[i][0] == 'Y' || argv[i][0] == 'y')
                    (*descriptors)[_SURF] = true;
                  else if(argv[i][0] == 'N' || argv[i][0] == 'n')
                    (*descriptors)[_SURF] = false;
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_DEPTH:                                   //  Incoming file path
                  (*depthFilenameLen) = strlen(argv[i]);
                  if(((*depthFilename) = (char*)malloc(((*depthFilenameLen) + 1) * sizeof(char))) == NULL)
                    cout << "ERROR: Unable to allocate depth file path string." << endl;
                  sprintf((*depthFilename), "%s", argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;

                case PARAM_E:                                       //  Incoming erosion size
                  (*erosion) = (unsigned char)atoi(argv[i]);
                  argtarget = PARAM_NONE;                           //  Reset argument target
                  break;
              }
          }

        i++;
      }

    return;
  }

/* Explain usage of this program and its options to the user. */
void usage(void)
  {
    cout << "Usage:  ./extract <options, each preceded by a flag>" << endl;
    cout << " e.g.:  ./extract -Q transformed/control_panel/transformed.270.png -BRISK N -ORB N -SIFT Y -SURF N -e 5 -SIFTcfg sift.dat" << endl;
    cout << "                  -mask masks/control_panel/mask.270.png" << endl;
    cout << "Flags:  -Q             following argument is the image to scan for features." << endl;
    cout << "        -BRISK         following argument is Y or N, enabling and disabling BRISK detection respectively." << endl;
    cout << "        -ORB           following argument is Y or N, enabling and disabling ORB detection respectively." << endl;
    cout << "        -SIFT          following argument is Y or N, enabling and disabling SIFT detection respectively." << endl;
    cout << "        -SURF          following argument is Y or N, enabling and disabling SURF detection respectively." << endl;
    cout << "        -mask          following argument is the file path for a mask image file." << endl;
    cout << "        -D             following argument is the file path for a depth buffer." << endl;
    cout << "        -e             following int argument sets the number of pixels by which to erode the unmasked edge." << endl;
    cout << "                       (Readme explains when this would be appropriate.)" << endl;
    cout << "        -bmeth         following argument is a string indicating which blur method to use on the query image." << endl;
    cout << "                       {GAUSS, BOX, MED} BOX is default." << endl;
    cout << "        -bkernel       following argument is the size of the blur filter. Default is 0 = no blur." << endl;
    cout << "        -showFeatures  will generate an image and a point cloud of detected features." << endl;
    cout << "        -showMask      will generate an image of the mask applied to the source." << endl;
    cout << "        -v             enable verbosity" << endl;
    cout << "        -?" << endl;
    cout << "        -help" << endl;
    cout << "        --help         displays this message." << endl;
    return;
  }