/* Author: Eric C Joyce
           Stevens Institute of Technology
   A small, interactive program that facilitates the Python script, refine_object_recog_dataset.py. 
   This program opens the given image in a clickable window and overlays all training/validation set
   object detections. Mouse clicks toggle whether detections are on or off.
   Press any key to close the window.
   Upon closing the window, ON/OFF states for each detection are reported back to the parent script,
   which updates the dataset.
*/
#include <iostream>
#include <opencv2/imgproc.hpp>                                      //  For annotating our GUI.
#include <opencv2/highgui.hpp>                                      //  For the display window.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
#define __CLICKER_DEBUG 1
*/

using namespace std;
using namespace cv;

#define PARAM_NONE          0                                       /* Flag indicating that no argument is to follow. */
#define PARAM_IMG           1                                       /* Flag indicating that the image file path is to follow. */
#define PARAM_DETECT_ULX    2                                       /* Flag indicating that a detection upper-left X is to follow. */
#define PARAM_DETECT_ULY    3                                       /* Flag indicating that a detection upper-left Y is to follow. */
#define PARAM_DETECT_LRX    4                                       /* Flag indicating that a detection lower-right X is to follow. */
#define PARAM_DETECT_LRY    5                                       /* Flag indicating that a detection lower-right Y is to follow. */

/********************************************************************
 Typedefs  */

typedef struct DetectionType
  {
    char* label;
    unsigned char labelLen;

    unsigned int upperLeftX;
    unsigned int upperLeftY;

    unsigned int lowerRightX;
    unsigned int lowerRightY;

    bool active;
  } Detection;

typedef struct ParamsType
  {
    char* filename;
    unsigned int filenameLen;
  } Params;

/********************************************************************
 Globals  */

Mat src;                                                            //  The (original) image.
Detection* detections;                                              //  Array of bounding boxes and labels received from command line.
unsigned int numDetections;                                         //  Length of that array.

/********************************************************************
 Prototypes  */

void drawDetections(Mat*);
void onMouseClick(int, int, int, int, void*);                       //  Mouse-click callback function.
//void drawReferenceCircle(Mat&, Point2i);                            //  Show where we've placed a reference.
bool parseArgv(int, char**, Params*);
void usage(void);

/********************************************************************
 Functions  */

int main(int argc, char** argv)
  {
    Mat img;                                                        //  The annotated image.
    unsigned int i;
    Params* params;                                                 //  Packaged run-time parameters.

    if((params = (Params*)malloc(sizeof(Params))) == NULL)          //  Allocate parameters structure.
      {
        cout << "ERROR: Unable to allocate parameter structure." << endl;
        return 1;
      }

    if(!parseArgv(argc, argv, params))                              //  Collect parameters and detections.
      {
        cout << "ERROR: Unable to parse command-line parameters." << endl;
        return 1;
      }

    if(params->filenameLen == 0)                                    //  Display usage if arguments were lacking.
      {
        for(i = 0; i < numDetections; i++)
          {
            if(detections[i].labelLen > 0)
              free(detections[i].label);
          }

        free(params);
        usage();
        return 0;
      }

    src = imread(params->filename, CV_LOAD_IMAGE_COLOR);            //  Read the image file indicated
    if(!src.data)                                                   //  Broken?
      {
        cout << "ERROR: could not open or find the image \"" << params->filename << "\"." << endl;
        if(params->filenameLen > 0)
          free(params->filename);
        for(i = 0; i < numDetections; i++)
          {
            if(detections[i].labelLen > 0)
              free(detections[i].label);
          }
        free(params);
        return 1;
      }

    drawDetections(&img);                                           //  Draw detection overlays.

    namedWindow("Detections", WINDOW_AUTOSIZE);                     //  Build a window containig the image.

    setMouseCallback("Detections", onMouseClick, (void*)&img);      //  Attach a mouse callback function.

    for(;;)
      {
        imshow("Detections", img);                                  //  Show/Refresh the window.
        if(waitKey(15) != -1)                                       //  Loop until user hits a key.
          break;
      }

    destroyWindow("Detections");                                    //  Finally tear down the info-collection window.

    for(i = 0; i < numDetections; i++)                              //  Produce the output that the parent script picks up.
      {
        if(detections[i].active)
          cout << "1";
        else
          cout << "0";
      }

    if(params->filenameLen > 0)                                     //  Clean up, go home.
      free(params->filename);
    for(i = 0; i < numDetections; i++)
      {
        if(detections[i].labelLen > 0)
          free(detections[i].label);
      }

    free(params);

    return 0;
  }

/* Render the active detections in green and the inactive ones in red. */
void drawDetections(Mat* img)
  {
    unsigned int i;
    Point2i upperleft;
    Point2i upperleft_label;
    Point2i lowerright;

    (*img) = src.clone();                                           //  Restore from source.

    for(i = 0; i < numDetections; i++)
      {
        upperleft = Point2i(detections[i].upperLeftX, detections[i].upperLeftY);
        upperleft_label = Point2i(detections[i].upperLeftX + 5, detections[i].upperLeftY + 30);
        lowerright = Point2i(detections[i].lowerRightX, detections[i].lowerRightY);

        if(detections[i].active)
          {
            rectangle((*img), upperleft, lowerright, Scalar(0, 255, 0));
            putText((*img), detections[i].label, upperleft_label, FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 255, 0), 1.8);
          }
        else
          {
            rectangle((*img), upperleft, lowerright, Scalar(0, 0, 255));
            putText((*img), detections[i].label, upperleft_label, FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 1.8);
          }
      }

    return;
  }

/* Respond to mouse clicks and update detections. */
void onMouseClick(int event, int x, int y, int flags, void* param)
  {
    unsigned int least_index = UINT_MAX;                            //  Find the intersected box with the smallest area.
    unsigned int least_area = UINT_MAX;
    unsigned int area;
    unsigned int i;
    unsigned int* indices;
    unsigned int intersections = 0;
    Mat& img = *(Mat*)param;                                        //  Recover the working-copy image.

    if(event == EVENT_LBUTTONDOWN)                                  //  Left-click
      {
        for(i = 0; i < numDetections; i++)
          {
            if( x < detections[i].lowerRightX && x > detections[i].upperLeftX &&
                y < detections[i].lowerRightY && y > detections[i].upperLeftY )
              intersections++;
          }
        if(intersections > 0)
          {
            if((indices = (unsigned int*)malloc(intersections * sizeof(int))) == NULL)
              {
                cout << "ERROR: Unable to allocate click-intersection array." << endl;
                exit(1);
              }
          }
        intersections = 0;
        for(i = 0; i < numDetections; i++)
          {
            if( x < detections[i].lowerRightX && x > detections[i].upperLeftX &&
                y < detections[i].lowerRightY && y > detections[i].upperLeftY )
              {
                indices[intersections] = i;
                intersections++;
              }
          }
        i = 0;
        while(i < intersections)
          {
            area = (detections[ indices[i] ].lowerRightX - detections[ indices[i] ].upperLeftX) *
                   (detections[ indices[i] ].lowerRightY - detections[ indices[i] ].upperLeftY);
            if(area < least_area)
              {
                least_index = indices[i];
                least_area = area;
              }
            i++;
          }

        if(least_index < UINT_MAX)                                  //  If an index was identified, flip it.
          detections[least_index].active = !detections[least_index].active;

        if(intersections > 0)
          free(indices);

        drawDetections(&img);                                       //  Refresh the display.
      }

    return;
  }

/* Parse the command-line arguments and set values accordingly */
bool parseArgv(int argc, char** argv, Params* params)
  {
    unsigned int i = 1;
    unsigned int j;
    char buffer[256];
    unsigned char tmp;
    unsigned char argtarget = PARAM_NONE;
                                                                    //  Initialize parameters.
    params->filenameLen = 0;                                        //  Initially, no image.
    numDetections = 0;                                              //  Initially, no detections.

    while(i < (unsigned int)argc)
      {
        if(strcmp(argv[i], "-img") == 0)                            //  String to follow is the image.
          argtarget = PARAM_IMG;
        else                                                        //  Not one of our flags... react to one of the flags.
          {
            switch(argtarget)
              {
                case PARAM_IMG:                                     //  Incoming image path string
                  params->filenameLen = (unsigned char)sprintf(buffer, "%s", argv[i]);
                  if((params->filename = (char*)malloc((params->filenameLen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < params->filenameLen; j++)
                    params->filename[j] = buffer[j];
                  params->filename[j] = '\0';

                  argtarget = PARAM_NONE;                           //  Reset argument target.
                  break;

                case PARAM_NONE:                                    //  Nothing flagged: this is the first part of a detection.
                  numDetections++;                                  //  Increment the number of detections.
                  if(numDetections == 1)
                    {
                      if((detections = (Detection*)malloc(sizeof(Detection))) == NULL)
                        {
                          cout << "ERROR: Unable to allocate detections array." << endl;
                          return false;
                        }
                    }
                  else
                    {
                      if((detections = (Detection*)realloc(detections, numDetections * sizeof(Detection))) == NULL)
                        {
                          cout << "ERROR: Unable to allocate detections array." << endl;
                          return false;
                        }
                    }
                  detections[numDetections - 1].labelLen = (unsigned char)sprintf(buffer, "%s", argv[i]);
                  if((detections[numDetections - 1].label =
                      (char*)malloc((detections[numDetections - 1].labelLen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < detections[numDetections - 1].labelLen; j++)
                    detections[numDetections - 1].label[j] = buffer[j];
                                                                    //  Null-terminate the string and set to active.
                  detections[numDetections - 1].label[j] = '\0';
                  detections[numDetections - 1].active = true;

                  argtarget = PARAM_DETECT_ULX;                     //  Advance the argument target to upper-left X.
                  break;

                case PARAM_DETECT_ULX:                              //  Incoming upper-left X value.
                  detections[numDetections - 1].upperLeftX = (unsigned int)atoi(argv[i]);

                  argtarget = PARAM_DETECT_ULY;                     //  Advance the argument target to upper-left X.
                  break;

                case PARAM_DETECT_ULY:                              //  Incoming upper-left Y value.
                  detections[numDetections - 1].upperLeftY = (unsigned int)atoi(argv[i]);

                  argtarget = PARAM_DETECT_LRX;                     //  Advance the argument target to upper-left X.
                  break;

                case PARAM_DETECT_LRX:                              //  Incoming lower-right X value.
                  detections[numDetections - 1].lowerRightX = (unsigned int)atoi(argv[i]);

                  argtarget = PARAM_DETECT_LRY;                     //  Advance the argument target to upper-left X.
                  break;

                case PARAM_DETECT_LRY:                              //  Incoming lower-right Y value.
                  detections[numDetections - 1].lowerRightY = (unsigned int)atoi(argv[i]);

                  argtarget = PARAM_NONE;                           //  Reset argument target.
                  break;
              }
          }

        i++;
      }

    return true;
  }

void usage(void)
  {
    cout << "CLICKER: Allow a user to keep or reject object detections ground-truths from training and validation sets." << endl;
    cout << endl;
    cout << "e.g.  ./clicker -img Enactment2/Users/vr1/POV/NormalViewCameraFrames/1267_126.7.png" << endl;
    cout << "                TransferFeederBox_Open 0 0 1145 676" << endl;
    cout << "                Disconnect_Closed 676 204 708 289" << endl;
    cout << "                Disconnect_Closed 657 319 696 377" << endl;
    cout << "                Disconnect_Unknown 873 294 919 432" << endl;
    cout << "                Disconnect_Closed 763 307 803 437" << endl;
    cout << endl;
    cout << "Flags:  -img    REQUIRED: the path to an image file." << endl;
    return;
  }
