/* Author: Eric C Joyce
           Stevens Institute of Technology
   A small, interactive program that facilitates the Python script, edit_mask_artifacts.py.
   This program opens the given mask, lays it over the given image in a clickable window,
   and draws a bounding box.
   By drawing your own boxes, you can include/exclude parts of the mask.
   Press any key to close the window.
   Upon closing the window, the revised mask and *_props.txt file are updated.
   This had to be done because there were FAR too many artifacts left in the source color maps.
*/
#include <iostream>
#include <opencv2/imgproc.hpp>                                      //  For annotating our GUI.
#include <opencv2/highgui.hpp>                                      //  For the display window.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
#define __EDIT_ARTIFACTS_DEBUG 1
*/

using namespace std;
using namespace cv;

#define PARAM_NONE          0                                       /* Flag indicating that no argument is to follow. */
#define PARAM_MASK          1                                       /* Flag indicating that the mask file path is to follow. */
#define PARAM_IMG           2                                       /* Flag indicating that the image file path is to follow. */

#define PEN_EXCLUDE  0                                              /* Boxes drawn when penType == PEN_EXCLUDE exlude areas from the mask (black). */
#define PEN_INCLUDE  1                                              /* Boxes drawn when penType == PEN_INCLUDE include areas in the mask (white). */

/********************************************************************
 Typedefs  */

typedef struct ZoneType
  {
    unsigned int upper_left_x;
    unsigned int upper_left_y;

    unsigned int lower_right_x;
    unsigned int lower_right_y;

    bool enabled;                                                   //  True: let mask here = 255; False: let mask here = 0.
  } Zone;

typedef struct ParamsType
  {
    char* mask_filename;
    unsigned int mask_filenameLen;

    char* img_filename;
    unsigned int img_filenameLen;

    bool helpme;
  } Params;

/********************************************************************
 Globals  */

Mat mask_src;                                                       //  The (original) mask.
Mat img_src;                                                        //  The (original) image.
unsigned int start_x, start_y;                                      //  Touchdown point for a new box.
unsigned int current_x, current_y;                                  //  Saved while moving.
bool drawing;                                                       //  Whether the pen is down.

Zone* zones;                                                        //  Global array of zones.
unsigned int zonesLen;

unsigned char penType;                                              //  Whether we are including or excluding.

/********************************************************************
 Prototypes  */

void draw(Mat*);
void onMouse(int, int, int, int, void*);                            //  Mouse callback function.
bool parseArgv(int, char**, Params*);
void usage(void);

/********************************************************************
 Functions  */

int main(int argc, char** argv)
  {
    Mat img;                                                        //  The annotated image. (Working copy).
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
                                                                    //  Display usage if arguments were lacking.
    if(params->mask_filenameLen == 0 || params->img_filenameLen == 0 || params->helpme)
      {
        if(params->mask_filename > 0)
          free(params->mask_filename);
        if(params->img_filename > 0)
          free(params->img_filename);
        free(params);
        usage();
        return 0;
      }

    mask_src = imread(params->mask_filename, IMREAD_GRAYSCALE);     //  Read the mask file indicated
    if(!mask_src.data)                                              //  Broken?
      {
        cout << "ERROR: could not open or find the mask \"" << params->mask_filename << "\"." << endl;
        if(params->mask_filename > 0)
          free(params->mask_filename);
        if(params->img_filename > 0)
          free(params->img_filename);
        free(params);
        return 1;
      }

    img_src = imread(params->img_filename, IMREAD_UNCHANGED);       //  Read the img file indicated
    if(!img_src.data)                                               //  Broken?
      {
        cout << "ERROR: could not open or find the image \"" << params->img_filename << "\"." << endl;
        if(params->mask_filename > 0)
          free(params->mask_filename);
        if(params->img_filename > 0)
          free(params->img_filename);
        free(params);
        return 1;
      }

    zonesLen = 0;                                                   //  Initialize global Zone counter.
    penType = PEN_EXCLUDE;                                          //  Initially we EXCLUDE.
    drawing = false;                                                //  Initially, we are not drawing.

    draw(&img);                                                     //  Draw.
    namedWindow("Mask Overlay", WINDOW_AUTOSIZE);                   //  Build a window containig the image.
    setMouseCallback("Mask Overlay", onMouse, (void*)&img);         //  Attach a mouse callback function.

    for(;;)
      {
        imshow("Mask Overlay", img);                                //  Show/Refresh the window.
        if(waitKey(15) != -1)                                       //  Loop until user hits a key.
          break;
      }

    destroyWindow("Mask Overlay");                                  //  Finally tear down the info-collection window.

    if(zonesLen > 0)                                                //  Report zones back to Python.
      {
        cout << +zonesLen;
        for(i = 0; i < zonesLen; i++)
          {
            if(zones[i].enabled)
              cout << "|" << +zones[i].upper_left_x << "," << +zones[i].upper_left_y << ";"
                          << +zones[i].lower_right_x << "," << +zones[i].lower_right_y << ";+";
            else
              cout << "|" << +zones[i].upper_left_x << "," << +zones[i].upper_left_y << ";"
                          << +zones[i].lower_right_x << "," << +zones[i].lower_right_y << ";-";
          }
      }
    else                                                            //  Report no zones.
      cout << "0";

    if(params->mask_filenameLen > 0)                                //  Clean up, go home.
      free(params->mask_filename);
    if(params->img_filenameLen > 0)
      free(params->img_filename);
    free(params);

    if(zonesLen > 0)
      free(zones);

    return 0;
  }

/* Render the cumulative effect. */
void draw(Mat* img)
  {
    Mat affect;                                                     //  Binary image.
    Mat tmp_mask;
    Mat working_mask;
    Mat mask_rgb;                                                   //  RGB conversion of the mask.
    vector<Point> locations;

    Point2i upperleft;
    Point2i lowerright;

    unsigned int i, x, y;

    (*img) = img_src.clone();                                       //  Restore from source.
    working_mask = mask_src.clone();                                //  Restore from source.

    findNonZero(mask_src, locations);                               //  Compute this once.

    for(i = 0; i < zonesLen; i++)                                   //  Apply the effect of each Zone to the mask.
      {
        if(zones[i].enabled)                                        //  Start with zeros and carve out an inclusion area
          {                                                         //    --ONLY where original mask pixels existed!!
            affect = Mat::zeros(Size(mask_src.cols, mask_src.rows), CV_8UC1);
            for(y = zones[i].upper_left_y; y < zones[i].lower_right_y; y++)
              {
                for(x = zones[i].upper_left_x; x < zones[i].lower_right_x; x++)
                  affect.at<uchar>(y, x) = 255;
              }
            bitwise_and(mask_src, affect, tmp_mask);
            bitwise_or(working_mask, tmp_mask, working_mask);       //  Apply.
          }
        else                                                        //  Start with 255s and carve out an exclusion zone.
          {
            affect = Mat(Size(mask_src.cols, mask_src.rows), CV_8UC1, Scalar(255));
            for(y = zones[i].upper_left_y; y < zones[i].lower_right_y; y++)
              {
                for(x = zones[i].upper_left_x; x < zones[i].lower_right_x; x++)
                  affect.at<uchar>(y, x) = 0;
              }
            bitwise_and(working_mask, affect, working_mask);        //  Apply.
          }
      }

    cvtColor(working_mask, mask_rgb, COLOR_GRAY2RGB);               //  Convert working copy.

    addWeighted((*img), 0.5, mask_rgb, 1.0, 0.0, (*img));           //  Weighted overlay.

    if(drawing)
      {
        if(current_x > start_x && current_y < start_y)
          {
            upperleft = Point2i(start_x, current_y);
            lowerright = Point2i(current_x, start_y);
          }
        else if(current_x < start_x && current_y < start_y)
          {
            upperleft = Point2i(current_x, current_y);
            lowerright = Point2i(start_x, start_y);
          }
        else if(current_x > start_x && current_y > start_y)
          {
            upperleft = Point2i(start_x, start_y);
            lowerright = Point2i(current_x, current_y);
          }
        else
          {
            upperleft = Point2i(current_x, start_y);
            lowerright = Point2i(start_x, current_y);
          }

        if(penType == PEN_EXCLUDE)                                  //  RED for EXCLUSION.
          rectangle((*img), upperleft, lowerright, Scalar(0, 0, 255));
        else if(penType == PEN_INCLUDE)                             //  GREEN for INCLUSION.
          rectangle((*img), upperleft, lowerright, Scalar(0, 255, 0));
      }

/*
    if(bogie)
      bitwise_not((*img), (*img));                                  //  Invert image to indicate.
*/
    return;
  }

/* Respond to mouse actions and update mask data. */
void onMouse(int event, int x, int y, int flags, void* param)
  {
    Mat& img = *(Mat*)param;                                        //  Recover the working-copy image.

    Point2i upperleft;
    Point2i lowerright;

    if(event == EVENT_LBUTTONDOWN)                                  //  Left mouse down.
      {
        start_x = x;
        start_y = y;

        current_x = x;
        current_y = y;

        drawing = true;
      }
    else if(event == EVENT_LBUTTONUP)                               //  Left mouse up.
      {
        drawing = false;

        current_x = x;
        current_y = y;

        if(current_x > start_x && current_y < start_y)
          {
            upperleft = Point2i(start_x, current_y);
            lowerright = Point2i(current_x, start_y);
          }
        else if(current_x < start_x && current_y < start_y)
          {
            upperleft = Point2i(current_x, current_y);
            lowerright = Point2i(start_x, start_y);
          }
        else if(current_x > start_x && current_y > start_y)
          {
            upperleft = Point2i(start_x, start_y);
            lowerright = Point2i(current_x, current_y);
          }
        else
          {
            upperleft = Point2i(current_x, start_y);
            lowerright = Point2i(start_x, current_y);
          }

        if(++zonesLen == 1)
          {
            if((zones = (Zone*)malloc(sizeof(Zone))) == NULL)
              {
                cout << "ERROR: Unable to allocate zones array." << endl;
                exit(1);
              }
          }
        else
          {
            if((zones = (Zone*)realloc(zones, zonesLen * sizeof(Zone))) == NULL)
              {
                cout << "ERROR: Unable to re-allocate zones array." << endl;
                exit(1);
              }
          }

        zones[zonesLen - 1].upper_left_x = upperleft.x;
        zones[zonesLen - 1].upper_left_y = upperleft.y;

        zones[zonesLen - 1].lower_right_x = lowerright.x;
        zones[zonesLen - 1].lower_right_y = lowerright.y;

        if(penType == PEN_EXCLUDE)
          zones[zonesLen - 1].enabled = false;
        else if(penType == PEN_INCLUDE)
          zones[zonesLen - 1].enabled = true;
      }
    else if(event == EVENT_MOUSEMOVE && drawing)                    //  Mouse move.
      {
        current_x = x;
        current_y = y;
      }
    else if(event == EVENT_RBUTTONUP)                               //  Right mouse up.
      {
        if(penType == PEN_EXCLUDE)                                  //  Change pen type.
          penType = PEN_INCLUDE;
        else if(penType == PEN_INCLUDE)
          penType = PEN_EXCLUDE;
      }

    draw(&img);                                                 //  Refresh the display.

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
    params->mask_filenameLen = 0;                                   //  Initially, no mask.
    params->img_filenameLen = 0;                                    //  Initially, no image.
    params->helpme = false;                                         //  Initially, no help.

    while(i < (unsigned int)argc)
      {
        if(strcmp(argv[i], "-mask") == 0)                           //  String to follow is the mask.
          argtarget = PARAM_MASK;
        else if(strcmp(argv[i], "-img") == 0)                       //  String to follow is the image.
          argtarget = PARAM_IMG;
        else if(strcmp(argv[i], "-?") == 0 || strcmp(argv[i], "-help") == 0)
          {
            params->helpme = true;
            argtarget = PARAM_NONE;
          }
        else                                                        //  Not one of our flags... react to one of the flags.
          {
            switch(argtarget)
              {
                case PARAM_MASK:                                    //  Incoming mask path string
                  params->mask_filenameLen = (unsigned char)sprintf(buffer, "%s", argv[i]);
                  if((params->mask_filename = (char*)malloc((params->mask_filenameLen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < params->mask_filenameLen; j++)
                    params->mask_filename[j] = buffer[j];
                  params->mask_filename[j] = '\0';

                  argtarget = PARAM_NONE;                           //  Reset argument target.
                  break;

                case PARAM_IMG:                                     //  Incoming image path string
                  params->img_filenameLen = (unsigned char)sprintf(buffer, "%s", argv[i]);
                  if((params->img_filename = (char*)malloc((params->img_filenameLen + 1) * sizeof(char))) == NULL)
                    return false;
                  for(j = 0; j < params->img_filenameLen; j++)
                    params->img_filename[j] = buffer[j];
                  params->img_filename[j] = '\0';

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
    cout << "EDIT_ARTIFACTS: A small, interactive program that facilitates the Python script, edit_mask_artifacts.py." << endl;
    cout << "                This program opens the given mask, lays it over the given image in a clickable window," << endl;
    cout << "                and draws a bounding box." << endl;
    cout << "                By drawing your own boxes, you can include/exclude parts of the mask." << endl;
    cout << "                Press any key to close the window." << endl;
    cout << "                Upon closing the window, the revised mask and *_props.txt file are updated." << endl;
    cout << endl;
    cout << "e.g.  ./edit_artifacts -mask BackBreaker1/GT/mask_0.png -img BackBreaker1/Users/vr1/POV/NormalViewCameraFrames/0_0.png" << endl;
    cout << endl;
    cout << "Flags:  -mask  REQUIRED: the path to a mask file." << endl;
    cout << "        -img   REQUIRED: the path to an image file." << endl;
    cout << endl;
    cout << "        -help  " << endl;
    cout << "        -?     Display this message." << endl;
    return;
  }
