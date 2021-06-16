/*  Eric C. Joyce, Stevens Institute of Technology, 2020.

    Make use of whichever descriptors are in the signature files we read.
    Build a KD-tree from the signatures and/or from the query image.
    Survey matches between query image features and the features of all objects.
    Matches are based on a threshold controlled by a run-time parameter.
    A match between a query image feature and a nearest neighbor in an object casts a vote for that object.
    Identify the top 'k' (again a run-time parameter) objects.
    The top 'k' objects go onto pose estimation.
    Depending on run-time parameters, each estimate is made either from a fresh round of correspondences between
    the query image and ONLY the signature for that object, OR using only those correspondences that originally
    contributed to the object's votes.
    Return as your final estimate the pose with the most inliers.
*/

/*
./pnp -Q DSLR_Photos/DSC09935.JPG -K cameras/SONY-DSLR-A580-Landscape-18mm.dat -sig signatures/single_view/SIFT/ -down 0.5 -iter 2000 -conf 0.9999 -ratio 0.9 -mmeth MUT -showFeatures -showInliers -nms 20.0
*/

#include <dirent.h>
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

#include "camera.h"
#include "descriptor.h"
#include "extractor.h"
#include "matcher.h"
#include "pnp_config.h"
#include "pnp_solver.h"
#include "signature.h"

/*
#define __PNP_DEBUG 1
*/

/**************************************************************************************************
 Typedefs  */

/**************************************************************************************************
 Prototypes  */
void pnp_init_query(PnPConfig* config, char**, unsigned int*, unsigned int*);
void pnp_init_sig(unsigned int, char*, unsigned int*, char**, unsigned int**);
void pnp_main(char*, unsigned int, unsigned int, char*, char*, unsigned int*, unsigned int, PnPConfig*, char*, char**);

/* Main loads signatures, converts them to a byte-array and an array of sub-array lengths.
   Main loads the camera data and converts that to a byte-array.
   Main loads an image and converts it to a byte-array.
   Main ships all these byte-arrays to pnp_main(), which computes a pose and writes to an output buffer. */
int main(int argc, char** argv)
  {
    PnPConfig pnp_config(argc, argv);                               //  Load parameters into a config object
    Signature sig;
    Camera cam;
    char* buffer;                                                   //  General-purpose

    char* imgByteArr;                                               //  Query image byte array
    unsigned int w, h;
    char* KByteArr;                                                 //  Camera intrinsics and distortion byte array
    char* sigByteArr;                                               //  Byte array of all signatures, packed together
    unsigned int* sigLengths;                                       //  Array of lengths of each signature substring
    unsigned int sigLen;                                            //  Length of 'sigLengths' = number of signatures

    char* result;                                                   //  Output buffer

    char* sigFileString;                                            //  Super string (including NULLs) of all file names
    unsigned int* sigFileLengths;                                   //  Lengths of substrings

    if(pnp_config.helpme())                                         //  Just displaying help? Do it and quit.
      {
        pnp_config.paramUsage();
        return 0;
      }

    #ifdef __PNP_DEBUG
    pnp_config.display();
    #endif

    /***********************************************************************
    **  However we get the query image as a byte array                    **
    ***********************************************************************/
    pnp_init_query(&pnp_config, &imgByteArr, &w, &h);

    /***********************************************************************
    **  However we get the camera intrinsics and distortion coeffs.       **
    ***********************************************************************/
    pnp_config.K(&buffer);                                          //  Write camera filename into buffer
    cam.load(buffer);                                               //  Load camera file
    free(buffer);                                                   //  Dump the buffer
    cam.writeByteArray(&KByteArr);                                  //  Load camera stuff as byte array

    /************************************************************************
    **  Write concatenated super-string (including NULLs) of all           **
    **  Signature file names to 'sigFileString'. Write each sub-string's   **
    **  length to 'sigFileLengths', and return the number of Signatures    **
    **  as 'sigLen'.                                                       **
    ************************************************************************/
    sigLen = pnp_config.fetchSignatures(&sigFileString, &sigFileLengths);
    pnp_init_sig(sigLen, sigFileString, sigFileLengths, &sigByteArr, &sigLengths);

    /************************************************************************
    **  However it is we got the image as a byte array,                    **
    **  the camera data as a byte array,                                   **
    **  and the Signatures as (super-string, array-of-lengths, length)...  **
    **  pass them and the settings to the entry-point function:            **
    ************************************************************************/
    if(w > 0 && h > 0)
      pnp_main(imgByteArr, w, h, KByteArr, sigByteArr, sigLengths, sigLen, &pnp_config, NULL, &result);
    else
      {
        cout << "ERROR: Query image not found." << endl;
        free(KByteArr);
        free(sigByteArr);
        free(sigLengths);
        free(sigFileString);
        free(sigFileLengths);
        return 0;
      }

    if(strlen(result) > 0)
      {
        cout << result << endl;                                     //  Print result
        free(result);
      }

    free(imgByteArr);                                               //  Clean up, go home
    free(KByteArr);
    free(sigByteArr);
    free(sigLengths);
    free(sigFileString);
    free(sigFileLengths);

    return 0;
  }

/*  Convert config bytes to PnPConfig object.
    Convert image bytes into image (cv::Mat).
    Convert K bytes into K (cv::Mat) and distortion vector. Unpack signatures.
    Either convert Rt bytes into initial r t guess or set to zero-vectors.

    Extract features from query.  Match features in signatures.
    Estimate poses for top candidates. Write best estimate to 'resultBuffer'

    Write program output to 'resultBuffer'. */
void pnp_main(char* imgByteArr,                                     //  Byte array for the query image
              unsigned int w, unsigned int h,                       //  Query image width and height
              char* KByteArr,                                       //  K-matrix-and-distortion byte array
              char* sigByteArr,                                     //  Byte array for all signatures
              unsigned int* sigLengths,                             //  Array of lengths, one for each signature sub-array
              unsigned int len,                                     //  Number of signatures = length of 'sigLengths'
              PnPConfig* config,                                    //  Address of a configuration-object
              char* RtByteArr,                                      //  Byte array for a previously-computed pose (may be NULL)
              char** resultBuffer)
  {
    Signature** signatures;                                         //  Array of Signature pointers
    Camera cam;                                                     //  Camera created from byte array
    cv::Mat img;                                                    //  The query image (reconstructed from byte array)
    cv::Mat K;                                                      //  Intrinsic matrix created from Camera
    cv::Mat distortion;                                             //  Distortion coefficients created from Camera
    cv::Mat mask;                                                   //  The extraction mask (if we have one)
    char* maskfilename;

    char* byteBuffer;                                               //  Cache for loading byte arrays into objects
    unsigned int offset = 0;                                        //  Offset into 'sigBytesArr'
    unsigned int i, j, l;
    unsigned int recognized;                                        //  Index into signatures of the recognized object
    unsigned char k = 0;
    cv::Vec3b pixel;                                                //  Temp, used to reconstruct the query image

    Extractor* extractor;
    Matcher* matcher;
    unsigned int* top;                                              //  Array of indices into signatures
    unsigned int* votes;
    unsigned int topLen;                                            //  Length of that array
    unsigned char flags = 0;                                        //  Flags for Descriptors used in all Signatures; informs Extractor

    cv::Mat r;                                                      //  If we have initial guesses for pose
    cv::Mat t;                                                      //  store rotation vector and translation vector here
    std::vector<cv::Point2f> pt2;                                   //  2D points (in the query)
    std::vector<cv::Point3f> pt3;                                   //  3D points (in a Signature)
    std::vector<cv::Point2f> pt2Refined;                            //  Refined run, 2D points (in the query)
    std::vector<cv::Point3f> pt3Refined;                            //  Refined run, 3D points (in a Signature)
    PnPSolver* solver;
    unsigned int maxInliers = 0;                                    //  Most inliers found so far
    cv::Mat rBest;                                                  //  Best pose found so far
    cv::Mat tBest;
    std::vector<int> inliersBest;                                   //  Indices of points that made up the best pose
    std::vector<cv::Point2f> pt2Best;                               //  2D points that contribute to the best pose estimate
    std::vector<cv::Point3f> pt3Best;                               //  3D points that contribute to the best pose estimate

    cv::Mat R;                                                      //  Rotation matrix derived from the Rodrigues-vector
    cv::Mat RH_T;                                                   //  RIGHT-HAND 4x4 transform: position object relative to camera at (0, 0, 0)
    cv::Mat RH_invT;                                                //  RIGHT-HAND 4x4 transform: position camera relative to object at (0, 0, 0)
    cv::Mat R2L;                                                    //  Reflection transform
    cv::Mat LH_T;                                                   //  LEFT-HAND 4x4 transform: position object relative to camera at (0, 0, 0)
    cv::Mat LH_invT;                                                //  LEFT-HAND 4x4 transform: position camera relative to object at (0, 0, 0)
    char* bestObjectFilename;                                       //  Take note of the signature with the most inliers
    char buffer[2048];                                              //  Temp cache for a string so we can measure resultBuffer's allocation

    char* str;                                                      //  Auxiliary output file names (if required)
    ofstream outputfh;                                              //  Auxiliary output stream (if required)
    std::vector<int> inliers;                                       //  Auxiliary output: PnPSolver's internal vector
    cv::Mat outimg;                                                 //  Auxiliary output image for rendering inliers (if required)
    cv::Mat colormap;                                               //  Auxiliary output: color feature points, if we're rendering them
    float x, y, z;                                                  //  Auxiliary output: store coordinates for writing to PLY
    unsigned char red, green, blue;                                 //  Auxiliary output: store colors for writing to PLY

    if((signatures = (Signature**)malloc(len * sizeof(Signature*))) == NULL)
      {
        cout << "ERROR: Unable to allocate Signatures array." << endl;
        return;
      }

    for(i = 0; i < len; i++)                                        //  Read 'len' byte-arrays into 'signatures'
      {
        if((byteBuffer = (char*)malloc(sigLengths[i] * sizeof(char))) == NULL)
          {
            cout << "ERROR: Unable to allocate byte sub-array." << endl;
            return;
          }
        for(j = 0; j < sigLengths[i]; j++)
          byteBuffer[j] = sigByteArr[offset + j];
        signatures[i] = new Signature();
        signatures[i]->readByteArray(byteBuffer);
        flags |= signatures[i]->flags();                            //  Collect signature flags
        offset += sigLengths[i];
        free(byteBuffer);
      }

    img = cv::Mat(h, w, CV_8UC3);                                   //  Allocate image
    l = 0;                                                          //  Restore query image from byte-array
    for(i = 0; i < h; i++)
      {
        for(j = 0; j < w; j++)
          {
            pixel = cv::Vec3b(imgByteArr[l], imgByteArr[l + 1], imgByteArr[l + 2]);
            l += 3;
            img.at<cv::Vec3b>(i, j) = pixel;
          }
      }
    if(config->convertToGrayscale())                                //  Converting to grayscale?
      cv::cvtColor(img, img, CV_RGB2GRAY);
    if(config->downSample() < 1.0)                                  //  Resizing?
      cv::resize(img, img, Size(round(w * config->downSample()), round(h * config->downSample())));

    if(config->hasMask())
      {
        config->mask(&maskfilename);
        mask = cv::imread(maskfilename, IMREAD_GRAYSCALE);          //  Load mask
      }

    extractor = new Extractor(flags);                               //  Create an Extractor
    extractor->setBlurMethod(config->blurMethod());                 //  Use the blur method from the Config object
    extractor->setBlurKernelSize(config->blurKernelSize());         //  Use the blur kernel size from the Config object
    extractor->setDownSample(config->downSample());                 //  If we're down-sampling, then divide coordinates by reciprocal
    extractor->setFeatureLimit(config->qLimit());                   //  Clamp the number of features (per descriptor) according to Config obj
    extractor->setNonMaxSuppression(config->nonMaxSuppression());   //  Set the suppression radius
    extractor->setBRISKparams(config);                              //  Configure detectors the way we want
    extractor->setORBparams(config);
    extractor->setSIFTparams(config);
    extractor->setSURFparams(config);
    extractor->setRenderBlur(config->renderBlurred());              //  Render the blurred query image, according to the Config object
    extractor->setRenderDetections(config->renderFeatures());       //  Render the detected featurs,  according to the Config object
    extractor->setWriteDetections(config->writeDetected());         //  Write the detected features' (X, Y) positions to file
    extractor->initDetectors();                                     //  Initialize detectors
    if(config->hasMask())
      extractor->extract(img, mask);                                //  Extract (masked) features
    else
      extractor->extract(img);                                      //  Extract features

    cam.readByteArray(KByteArr);                                    //  Load intrinsics and distortion from byte array
    cam.K(&K);                                                      //  Separate K and distortion into cv::Mats
    cam.dist(&distortion);

    solver = new PnPSolver(K, distortion);                          //  Create the solver
    solver->setIterations(config->iterations());                    //  Use the number of iterations defined by the Config object
    solver->setReprojectionError(config->reprojectionError());      //  Use the reprojection error defined by the Config object
    solver->setConfidence(config->confidence());                    //  Use the confidence defined by the Config object
    solver->setMethod(config->ransacMethod());                      //  Use the method defined by the Config object

    matcher = new Matcher((*config));                               //  Build a Matcher with KD trees and Hash Tables as per the Config object
    if(config->matchMethod() == _MATCH_MODE_SIG2Q)
      {
        matcher->train(signatures, len);                            //  "Train" the matcher on the received Signatures
        matcher->match(extractor);                                  //  Match extracted features with Signature features
      }
    else if(config->matchMethod() == _MATCH_MODE_Q2SIG)
      {
        matcher->train(extractor);                                  //  "Train" the matcher on the Extractor-extracted features
        matcher->match(signatures, len);                            //  Match Signature features with Extractor-extracted features
      }
    else
      {
        matcher->train(extractor);                                  //  "Train" one matcher on the Extractor-extracted features,
        matcher->train(signatures, len);                            //  "train" another matcher on the received Signatures,
        matcher->match();                                           //  Match mutually
      }

    topLen = matcher->top(&top, &votes);                            //  Identify the top matched objects
    for(i = 0; i < topLen; i++)                                     //  For each of the top candidates...
      {
        #ifdef __PNP_DEBUG
        cout << "#" << +(i + 1) << ": ";
        signatures[ top[i] ]->printFilename();
        cout << ", " << +votes[i] << " votes" << endl;
        #endif

        if(config->poseEstMethod() == _POSE_EST_BY_VOTE)            //  Consider only those correspondences that prefer this object anyway
          l = matcher->correspondences(signatures[ top[i] ], extractor, &pt2, &pt3);
        else                                                        //  Make correspondences for ONLY this object
          {
            if(config->matchMethod() == _MATCH_MODE_SIG2Q)
              {
                matcher->clearSig();                                //  Clear the Signature indices only
                matcher->train(signatures + top[i], 1);             //  (Re)"Train" the Signature-matcher on a single Signature
                matcher->match(extractor);                          //  Match extracted features with Signature features
              }
            else if(config->matchMethod() == _MATCH_MODE_Q2SIG)
              matcher->match(signatures + top[i], 1);               //  Match Signature features with Extractor-extracted features
            else
              {
                matcher->clearSig();                                //  Clear the Signature indices only
                matcher->train(signatures + top[i], 1);             //  (Re)"Train" the Signature-matcher on a single Signature
                matcher->match();                                   //  Match mutually
              }

            l = matcher->correspondences(signatures[ top[i] ], extractor, &pt2, &pt3);
          }

        if(config->writeObjCorr())                                  //  Are we writing this object's correspondences?
          {
            str = signatures[ top[i] ]->fileStem();                 //  Name the file ./this/is/my/signature.correspondences
            if(str != NULL)                                         //  Write all (2D) --> (3D) as plain text
              {
                sprintf(buffer, "%s.correspondences", str);
                outputfh.open(buffer);
                for(j = 0; j < l; j++)
                  {
                    outputfh << "(" << pt2.at(j).x << ", " << pt2.at(j).y << ") --> ";
                    outputfh << "(" << pt3.at(j).x << ", " << pt3.at(j).y << ", " << pt3.at(j).z << ")" << endl;
                  }
                outputfh.close();
                free(str);
              }
          }

        if(l >= solver->minimum())                                  //  Enough to solve pose?
          {
            l = solver->solve(pt2, pt3);                            //  SOLVE !!

            if(config->renderInliers() && l > 0)                    //  Are we rendering inliers? Are there any inliers?
              {
                inliers = solver->inliers();                        //  Get a reference to the Solver object's inlier vector

                str = signatures[ top[i] ]->fileStem();             //  Name the file ./this/is/my/signature.inliers
                if(str != NULL)                                     //  Render inliers
                  {
                    sprintf(buffer, "%s.inliers.png", str);

                    outimg = img.clone();                           //  Copy the query image
                    if(!config->convertToGrayscale())
                      cv::cvtColor(outimg, outimg, CV_RGB2GRAY);    //  Convert to grayscale then back to color
                    cv::cvtColor(outimg, outimg, CV_GRAY2RGB);      //  so the features stand out

                    colormap = cv::Mat(cv::Size(l, 1), CV_8UC1);    //  Generate a color map with as many intervals
                    for(j = 0; j < l; j++)                          //  as there are inliers for this object
                      colormap.at<unsigned char>(j) = (unsigned char)round(255.0 * (float)j / (float)l );
                    applyColorMap(colormap, colormap, COLORMAP_JET);

                    for(j = 0; j < l; j++)                          //  For all inliers
                      {
                                                                    //  Tattoo output image
                        cv::circle(outimg, Point(pt2.at( inliers.at(j) ).x * config->downSample(),
                                                 pt2.at( inliers.at(j) ).y * config->downSample()), 3.0,
                                           Scalar(colormap.at<unsigned char>(j * 3),
                                                  colormap.at<unsigned char>(j * 3 + 1),
                                                  colormap.at<unsigned char>(j * 3 + 2)), 2, 8);
                      }
                    cv::imwrite(buffer, outimg);                    //  Write the tattooed image

                    sprintf(buffer, "%s.inliers.ply", str);         //  Additionally create a PLY file so we can see which points match which
                    outputfh.open(buffer);
                    outputfh << "ply" << endl;                      //  Ply format header and declare endianness
                    outputfh << "format binary_little_endian 1.0" << endl;
                    outputfh << "element vertex " << +l << endl;    //  Declare number of vertices
                    outputfh << "property float x" << endl;         //  Vertex properties are the same as in the original PLY
                    outputfh << "property float y" << endl;
                    outputfh << "property float z" << endl;
                    outputfh << "property uchar red" << endl;
                    outputfh << "property uchar green" << endl;
                    outputfh << "property uchar blue" << endl;
                    outputfh << "end_header" << endl;               //  Close the header
                    for(j = 0; j < l; j++)                          //  For all inliers
                      {
                        x = pt3.at( inliers.at(j) ).x;
                        y = pt3.at( inliers.at(j) ).y;
                        z = pt3.at( inliers.at(j) ).z;
                        red   = colormap.at<unsigned char>(j * 3 + 2);
                        green = colormap.at<unsigned char>(j * 3 + 1);
                        blue  = colormap.at<unsigned char>(j * 3);  //  Yes, this is intentional: OpenCV renders BGR.

                        outputfh.write((char*)(&x), sizeof(float));
                        outputfh.write((char*)(&y), sizeof(float));
                        outputfh.write((char*)(&z), sizeof(float));
                        outputfh.write((char*)(&red), sizeof(char));
                        outputfh.write((char*)(&green), sizeof(char));
                        outputfh.write((char*)(&blue), sizeof(char));
                      }
                    outputfh.close();

                    free(str);
                  }
              }

            if(config->writeObjInliers())                           //  Are we writing this object's inliers?
              {
                inliers = solver->inliers();                        //  Get a reference to the Solver object's inlier vector

                str = signatures[ top[i] ]->fileStem();             //  Name the file ./this/is/my/signature.inliers
                if(str != NULL)                                     //  Write all (2D) --> (3D) as plain text
                  {
                    sprintf(buffer, "%s.inliers", str);
                    outputfh.open(buffer);
                    for(j = 0; j < l; j++)
                      {
                        outputfh << "(" << pt2.at( inliers.at(j) ).x << ", " << pt2.at( inliers.at(j) ).y << ") --> ";
                        outputfh << "(" << pt3.at( inliers.at(j) ).x << ", " << pt3.at( inliers.at(j) ).y << ", " << pt3.at( inliers.at(j) ).z << ")" << endl;
                      }
                    outputfh.close();
                    free(str);
                  }
              }

            #ifdef __PNP_DEBUG
            cout << "  " << +l << " inliers" << endl;
            #endif

            if(l > maxInliers)                                      //  Better than before (or better than none)? Save!
              {
                recognized = top[i];                                //  Save the index into 'signatures'
                maxInliers = l;                                     //  Save the superlative number of inliers

                pt2Best.clear();
                pt3Best.clear();
                for(j = 0; j < pt2.size(); j++)                     //  pt2 and pt3 are in lock-step
                  {
                    pt2Best.push_back( Point2f(pt2.at(j).x, pt2.at(j).y) );
                    pt3Best.push_back( Point3f(pt3.at(j).x, pt3.at(j).y, pt3.at(j).z) );
                  }

                inliers = solver->inliers();                        //  Get a copy of the Solver's inliers
                inliersBest.clear();                                //  Clear the vector of best-so-far inliers
                for(j = 0; j < l; j++)                              //  Clone the Solver's inlier vector
                  inliersBest.push_back( inliers.at(j) );

                solver->rvec(&rBest);                               //  Save the superlative pose
                solver->tvec(&tBest);                               //  *** AS IS, t-vec IS THE 4th COLUMN OF THE EXTRINSICS !!!
                if(k > 0)                                           //  (Re)set the best file name
                  free(bestObjectFilename);
                k = signatures[ top[i] ]->filename(&bestObjectFilename);
              }
          }
      }

    if(maxInliers > 0)                                              //  Did we have a single pose with inliers?
      {
        if(config->refineEstimate())                                //  Are we re-running RANSAC on the best inliers only?
          {
            #ifdef __PNP_DEBUG
            cout << "Refining pose estimate using inliers" << endl;
            #endif

            if(maxInliers >= solver->minimum())                     //  Do we have enough inliers to re-estimate?
              {
                for(i = 0; i < maxInliers; i++)                     //  Reduce pt2 and pt3 to only their inliers
                  {
                    pt2Refined.push_back( Point2f(pt2Best.at( inliersBest.at(i) ).x,
                                                  pt2Best.at( inliersBest.at(i) ).y) );
                    pt3Refined.push_back( Point3f(pt3Best.at( inliersBest.at(i) ).x,
                                                  pt3Best.at( inliersBest.at(i) ).y,
                                                  pt3Best.at( inliersBest.at(i) ).z) );
                  }

                maxInliers = solver->solve(pt2Refined, pt3Refined); //  (Re)-SOLVE !!

                if(config->renderInliers() && maxInliers > 0)       //  Are we rendering inliers? Are there any inliers?
                  {
                    inliers = solver->inliers();                    //  Get a reference to the Solver object's inlier vector

                    str = signatures[ recognized ]->fileStem();     //  Name the file ./this/is/my/signature.inliers
                    if(str != NULL)                                 //  Render inliers
                      {
                        sprintf(buffer, "%s.inliers_refined.png", str);

                        outimg = img.clone();                       //  Copy the query image
                        if(!config->convertToGrayscale())
                          cv::cvtColor(outimg, outimg, CV_RGB2GRAY);//  Convert to grayscale then back to color
                        cv::cvtColor(outimg, outimg, CV_GRAY2RGB);  //  so the features stand out

                                                                    //  Generate a color map with as many intervals
                                                                    //  as there are inliers for this object
                        colormap = cv::Mat(cv::Size(maxInliers, 1), CV_8UC1);
                        for(j = 0; j < maxInliers; j++)
                          colormap.at<unsigned char>(j) = (unsigned char)round(255.0 * (float)j / (float)maxInliers );
                        applyColorMap(colormap, colormap, COLORMAP_JET);

                        for(j = 0; j < maxInliers; j++)             //  For all inliers
                          {
                                                                    //  Tattoo output image
                            cv::circle(outimg, Point(pt2Refined.at( inliers.at(j) ).x * config->downSample(),
                                                     pt2Refined.at( inliers.at(j) ).y * config->downSample()), 3.0,
                                               Scalar(colormap.at<unsigned char>(j * 3),
                                                      colormap.at<unsigned char>(j * 3 + 1),
                                                      colormap.at<unsigned char>(j * 3 + 2)), 2, 8);
                          }
                        cv::imwrite(buffer, outimg);                //  Write the tattooed image
                                                                    //  Additionally create a PLY file so we can see which points match which
                        sprintf(buffer, "%s.inliers_refined.ply", str);
                        outputfh.open(buffer);
                        outputfh << "ply" << endl;                  //  Ply format header and declare endianness
                        outputfh << "format binary_little_endian 1.0" << endl;
                                                                    //  Declare number of vertices
                        outputfh << "element vertex " << +maxInliers << endl;
                        outputfh << "property float x" << endl;     //  Vertex properties are the same as in the original PLY
                        outputfh << "property float y" << endl;
                        outputfh << "property float z" << endl;
                        outputfh << "property uchar red" << endl;
                        outputfh << "property uchar green" << endl;
                        outputfh << "property uchar blue" << endl;
                        outputfh << "end_header" << endl;           //  Close the header
                        for(j = 0; j < maxInliers; j++)             //  For all inliers
                          {
                            x = pt3Refined.at( inliers.at(j) ).x;
                            y = pt3Refined.at( inliers.at(j) ).y;
                            z = pt3Refined.at( inliers.at(j) ).z;
                            red   = colormap.at<unsigned char>(j * 3 + 2);
                            green = colormap.at<unsigned char>(j * 3 + 1);
                            blue  = colormap.at<unsigned char>(j * 3);
                                                                    //  Yes, this is intentional: OpenCV renders BGR.

                            outputfh.write((char*)(&x), sizeof(float));
                            outputfh.write((char*)(&y), sizeof(float));
                            outputfh.write((char*)(&z), sizeof(float));
                            outputfh.write((char*)(&red), sizeof(char));
                            outputfh.write((char*)(&green), sizeof(char));
                            outputfh.write((char*)(&blue), sizeof(char));
                          }
                        outputfh.close();

                        free(str);
                      }
                  }

                if(config->writeObjInliers())                       //  Are we writing this object's inliers?
                  {
                    inliers = solver->inliers();                    //  Get a reference to the Solver object's inlier vector

                    str = signatures[ recognized ]->fileStem();     //  Name the file ./this/is/my/signature.inliers
                    if(str != NULL)                                 //  Write all (2D) --> (3D) as plain text
                      {
                        sprintf(buffer, "%s.inliers_refined.txt", str);
                        outputfh.open(buffer);
                        for(j = 0; j < l; j++)
                          {
                            outputfh << "(" << pt2Refined.at( inliers.at(j) ).x << ", " << pt2Refined.at( inliers.at(j) ).y << ") -->";
                            outputfh << "(" << pt3Refined.at( inliers.at(j) ).x << ", " << pt3Refined.at( inliers.at(j) ).y << ", " << pt3Refined.at( inliers.at(j) ).z << ")" << endl;
                          }
                        outputfh.close();
                        free(str);
                      }
                  }

                solver->rvec(&rBest);                               //  Save the updated Rodrigues-vector
                solver->tvec(&tBest);                               //  Save the updated... 4th COLUMN OF THE EXTRINSICS !!!
              }
          }

        R = cv::Mat(3, 3, CV_64FC1);                                //  (R and t are both type CV_64F)
        cv::Rodrigues(rBest, R, cv::noArray());                     //  Convert Rodrigues-vector to Rotation Matrix

        RH_T = cv::Mat(4, 4, CV_32FC1);                             //  Make a 4x4: object relative to camera
        RH_T.at<float>(0, 0) = (float)(R.at<double>(0, 0));  RH_T.at<float>(0, 1) = (float)(R.at<double>(0, 1));  RH_T.at<float>(0, 2) = (float)(R.at<double>(0, 2));  RH_T.at<float>(0, 3) = (float)(tBest.at<double>(0));
        RH_T.at<float>(1, 0) = (float)(R.at<double>(1, 0));  RH_T.at<float>(1, 1) = (float)(R.at<double>(1, 1));  RH_T.at<float>(1, 2) = (float)(R.at<double>(1, 2));  RH_T.at<float>(1, 3) = (float)(tBest.at<double>(1));
        RH_T.at<float>(2, 0) = (float)(R.at<double>(2, 0));  RH_T.at<float>(2, 1) = (float)(R.at<double>(2, 1));  RH_T.at<float>(2, 2) = (float)(R.at<double>(2, 2));  RH_T.at<float>(2, 3) = (float)(tBest.at<double>(2));
        RH_T.at<float>(3, 0) = 0.0;                          RH_T.at<float>(3, 1) = 0.0;                          RH_T.at<float>(3, 2) = 0.0;                          RH_T.at<float>(3, 3) = 1.0;

        RH_invT = RH_T.inv();                                       //  Store inversion of right-handed T: camera relative to object

        R2L = cv::Mat_<float>::zeros(4, 4);                         //  Build the reflector
        R2L.at<float>(0, 0) = -1.0;
        R2L.at<float>(1, 2) =  1.0;
        R2L.at<float>(2, 1) =  1.0;
        R2L.at<float>(3, 3) =  1.0;

        LH_T = R2L * RH_T;                                          //  Left-handed obj rel to cam
        LH_invT = R2L * RH_invT;                                    //  Left-handed cam rel to obj

        i = sprintf(buffer, "RH_CamRelObj[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nRH_ObjRelCam[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nLH_CamRelObj[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nLH_ObjRelCam[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nInliers[%d]\nObject[%s]\n",
                             RH_invT.at<float>(0, 0), RH_invT.at<float>(0, 1), RH_invT.at<float>(0, 2), RH_invT.at<float>(0, 3),
                             RH_invT.at<float>(1, 0), RH_invT.at<float>(1, 1), RH_invT.at<float>(1, 2), RH_invT.at<float>(1, 3),
                             RH_invT.at<float>(2, 0), RH_invT.at<float>(2, 1), RH_invT.at<float>(2, 2), RH_invT.at<float>(2, 3),
                             RH_invT.at<float>(3, 0), RH_invT.at<float>(3, 1), RH_invT.at<float>(3, 2), RH_invT.at<float>(3, 3),

                             RH_T.at<float>(0, 0), RH_T.at<float>(0, 1), RH_T.at<float>(0, 2), RH_T.at<float>(0, 3),
                             RH_T.at<float>(1, 0), RH_T.at<float>(1, 1), RH_T.at<float>(1, 2), RH_T.at<float>(1, 3),
                             RH_T.at<float>(2, 0), RH_T.at<float>(2, 1), RH_T.at<float>(2, 2), RH_T.at<float>(2, 3),
                             RH_T.at<float>(3, 0), RH_T.at<float>(3, 1), RH_T.at<float>(3, 2), RH_T.at<float>(3, 3),

                             LH_invT.at<float>(0, 0), LH_invT.at<float>(0, 1), LH_invT.at<float>(0, 2), LH_invT.at<float>(0, 3),
                             LH_invT.at<float>(1, 0), LH_invT.at<float>(1, 1), LH_invT.at<float>(1, 2), LH_invT.at<float>(1, 3),
                             LH_invT.at<float>(2, 0), LH_invT.at<float>(2, 1), LH_invT.at<float>(2, 2), LH_invT.at<float>(2, 3),
                             LH_invT.at<float>(3, 0), LH_invT.at<float>(3, 1), LH_invT.at<float>(3, 2), LH_invT.at<float>(3, 3),

                             LH_T.at<float>(0, 0), LH_T.at<float>(0, 1), LH_T.at<float>(0, 2), LH_T.at<float>(0, 3),
                             LH_T.at<float>(1, 0), LH_T.at<float>(1, 1), LH_T.at<float>(1, 2), LH_T.at<float>(1, 3),
                             LH_T.at<float>(2, 0), LH_T.at<float>(2, 1), LH_T.at<float>(2, 2), LH_T.at<float>(2, 3),
                             LH_T.at<float>(3, 0), LH_T.at<float>(3, 1), LH_T.at<float>(3, 2), LH_T.at<float>(3, 3),

                             maxInliers, bestObjectFilename);
        if(((*resultBuffer) = (char*)malloc((i + 1) * sizeof(char))) == NULL)
          {
            cout << "ERROR: Unable to allocate result buffer." << endl;
            return;
          }
        sprintf((*resultBuffer), "RH_CamRelObj[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nRH_ObjRelCam[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nLH_CamRelObj[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nLH_ObjRelCam[%f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f,\n             %f, %f, %f, %f]\nInliers[%d]\nObject[%s]\n",
                                  RH_invT.at<float>(0, 0), RH_invT.at<float>(0, 1), RH_invT.at<float>(0, 2), RH_invT.at<float>(0, 3),
                                  RH_invT.at<float>(1, 0), RH_invT.at<float>(1, 1), RH_invT.at<float>(1, 2), RH_invT.at<float>(1, 3),
                                  RH_invT.at<float>(2, 0), RH_invT.at<float>(2, 1), RH_invT.at<float>(2, 2), RH_invT.at<float>(2, 3),
                                  RH_invT.at<float>(3, 0), RH_invT.at<float>(3, 1), RH_invT.at<float>(3, 2), RH_invT.at<float>(3, 3),

                                  RH_T.at<float>(0, 0), RH_T.at<float>(0, 1), RH_T.at<float>(0, 2), RH_T.at<float>(0, 3),
                                  RH_T.at<float>(1, 0), RH_T.at<float>(1, 1), RH_T.at<float>(1, 2), RH_T.at<float>(1, 3),
                                  RH_T.at<float>(2, 0), RH_T.at<float>(2, 1), RH_T.at<float>(2, 2), RH_T.at<float>(2, 3),
                                  RH_T.at<float>(3, 0), RH_T.at<float>(3, 1), RH_T.at<float>(3, 2), RH_T.at<float>(3, 3),

                                  LH_invT.at<float>(0, 0), LH_invT.at<float>(0, 1), LH_invT.at<float>(0, 2), LH_invT.at<float>(0, 3),
                                  LH_invT.at<float>(1, 0), LH_invT.at<float>(1, 1), LH_invT.at<float>(1, 2), LH_invT.at<float>(1, 3),
                                  LH_invT.at<float>(2, 0), LH_invT.at<float>(2, 1), LH_invT.at<float>(2, 2), LH_invT.at<float>(2, 3),
                                  LH_invT.at<float>(3, 0), LH_invT.at<float>(3, 1), LH_invT.at<float>(3, 2), LH_invT.at<float>(3, 3),

                                  LH_T.at<float>(0, 0), LH_T.at<float>(0, 1), LH_T.at<float>(0, 2), LH_T.at<float>(0, 3),
                                  LH_T.at<float>(1, 0), LH_T.at<float>(1, 1), LH_T.at<float>(1, 2), LH_T.at<float>(1, 3),
                                  LH_T.at<float>(2, 0), LH_T.at<float>(2, 1), LH_T.at<float>(2, 2), LH_T.at<float>(2, 3),
                                  LH_T.at<float>(3, 0), LH_T.at<float>(3, 1), LH_T.at<float>(3, 2), LH_T.at<float>(3, 3),

                                  maxInliers, bestObjectFilename);
      }
    else                                                            //  No pose estimate
      {
        i = sprintf(buffer, "NoPose\n");
        if(((*resultBuffer) = (char*)malloc((i + 1) * sizeof(char))) == NULL)
          {
            cout << "ERROR: Unable to allocate output buffer." << endl;
            return;
          }
        sprintf((*resultBuffer), "NoPose\n");
        (*resultBuffer)[i] = '\0';
      }

    if(topLen > 0)
      free(top);

    for(i = 0; i < len; i++)                                        //  Clean up, go home
      delete signatures[i];

    free(signatures);

    delete extractor;
    delete matcher;
    delete solver;

    return;
  }

/* Open a hard-coded file (just for now, for testing), convert it to a byte-array,
   and write the bytes and dimensions to the given arguments. */
void pnp_init_query(PnPConfig* config, char** byteArr, unsigned int* w, unsigned int* h)
  {
    cv::Mat img;
    cv::Vec3b pixel;
    unsigned int channels;
    unsigned int i, j, l;
    char* qfilename;

    config->Q(&qfilename);

    img = cv::imread(qfilename, CV_LOAD_IMAGE_COLOR);               //  Load image
    (*w) = img.cols;
    (*h) = img.rows;
    channels = img.channels();
                                                                    //  Allocate space for image byte array
    if(((*byteArr) = (char*)malloc((*w) * (*h) * channels * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate byte array for query image." << endl;
        return;
      }

    l = 0;                                                          //  Copy query image to byte array
    for(i = 0; i < (*h); i++)                                       //  Iterate over rows (Y)
      {
        for(j = 0; j < (*w); j++)                                   //  Iterate over columns (X)
          {
            pixel = img.at<cv::Vec3b>(i, j);
            (*byteArr)[l] = pixel[0];
            l++;
            (*byteArr)[l] = pixel[1];
            l++;
            (*byteArr)[l] = pixel[2];
            l++;
          }
      }

    return;
  }

/* Open the Signature files concatenated in 'sigFileString' and write them as a concatenated byte-array 'sigByteBuffer'.
   Write their sub-array lengths in 'sigLengthsBuffer'.
   'sigLen' does not change. Number of files = Number of Signatures = Number of sub-arrays. */
void pnp_init_sig(unsigned int sigLen, char* sigFileString, unsigned int* sigFileLengths,
                  char** sigByteBuffer, unsigned int** sigLengthsBuffer)
  {
    Signature sig;
    char* filename;                                                 //  Storage for a single Signature file name
    char* byteArr;                                                  //  Byte array for a single signature
    unsigned int offset = 0;                                        //  Offset into the array of all signature file names
    unsigned int len = 0;                                           //  Offset into the byte array
    unsigned int i, j;

                                                                    //  Allocate an array of lengths for each signature byte arrays
    if(((*sigLengthsBuffer) = (unsigned int*)malloc(sigLen * sizeof(int))) == NULL)
      {
        cout << "ERROR: Unable to allocate signature-byte lengths array." << endl;
        return;
      }

    //////////////////////////////////////////////////////////////////  This is a count-up, so we know how much to allocate for byte array
    for(i = 0; i < sigLen; i++)                                     //  For every signature...
      {
                                                                    //  Allocate space for the file name plus NULL
        if((filename = (char*)malloc((sigFileLengths[i] + 1) * sizeof(char))) == NULL)
          {
            cout << "ERROR: Unable to allocate file name " << +j << endl;
            return;
          }

        for(j = 0; j < sigFileLengths[i]; j++)                      //  Copy file name substring
          filename[j] = sigFileString[offset + j];
        filename[j] = '\0';

        offset += sigFileLengths[i] + 1;                            //  Advance to next file name substring

        sig.load(filename);                                         //  Load that signature
        (*sigLengthsBuffer)[i] = sig.writeByteArray(&byteArr);      //  Save byte-array length (never mind the bytes for now)
        len += (*sigLengthsBuffer)[i];                              //  Increase size of superstring length

        free(byteArr);
        free(filename);
      }
                                                                    //  Allocate signature super-string
    if(((*sigByteBuffer) = (char*)malloc(len * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate signatures' byte array." << endl;
        return;
      }
    offset = 0;                                                     //  Reset
    len = 0;                                                        //  Reset

    //////////////////////////////////////////////////////////////////  We've counted; now allocate and convert
    for(i = 0; i < sigLen; i++)                                     //  For every signature...
      {
                                                                    //  Allocate space for the file name plus NULL
        if((filename = (char*)malloc((sigFileLengths[i] + 1) * sizeof(char))) == NULL)
          {
            cout << "ERROR: Unable to allocate file name " << +j << endl;
            return;
          }

        for(j = 0; j < sigFileLengths[i]; j++)                      //  Copy file name substring
          filename[j] = sigFileString[offset + j];
        filename[j] = '\0';

        sig.load(filename);
        sig.setIndex(i);                                            //  This is the 'i'th object
        sig.writeByteArray(&byteArr);                               //  Save byte-array
        for(j = 0; j < (*sigLengthsBuffer)[i]; j++)
          (*sigByteBuffer)[len + j] = byteArr[j];
        len += (*sigLengthsBuffer)[i];
        free(byteArr);

        offset += sigFileLengths[i] + 1;

        free(filename);
      }

    return;
  }
