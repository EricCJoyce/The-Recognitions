#ifndef __CAMERA_CPP
#define __CAMERA_CPP

#include "camera.h"

/**************************************************************************************************
 Constructors  */

/* Camera constructor, no data given */
Camera::Camera()
  {
    unsigned int i;

    _fx = 0.0;
    _fy = 0.0;
    _s = 0.0;
    _cx = 0.0;
    _cy = 0.0;

    _k1 = 0.0;
    _k2 = 0.0;
    _p1 = 0.0;
    _p2 = 0.0;
    _k3 = 0.0;
    _k4 = 0.0;
    _k5 = 0.0;
    _k6 = 0.0;

    _s1 = 0.0;
    _s2 = 0.0;
    _s3 = 0.0;
    _s4 = 0.0;

    _tx = 0.0;
    _ty = 0.0;

    for(i = 0; i < CAMERA_HEADER_LENGTH; i++)
      _header[i] = '\0';
  }

/* Camera constructor, file path given */
Camera::Camera(char* filename)
  {
    unsigned int i;

    _fx = 0.0;                                                      //  Set these to zero in case
    _fy = 0.0;                                                      //  reading from file fails
    _s = 0.0;
    _cx = 0.0;
    _cy = 0.0;

    _k1 = 0.0;
    _k2 = 0.0;
    _p1 = 0.0;
    _p2 = 0.0;
    _k3 = 0.0;
    _k4 = 0.0;
    _k5 = 0.0;
    _k6 = 0.0;

    _s1 = 0.0;
    _s2 = 0.0;
    _s3 = 0.0;
    _s4 = 0.0;

    _tx = 0.0;
    _ty = 0.0;

    for(i = 0; i < CAMERA_HEADER_LENGTH; i++)
      _header[i] = '\0';

    load(filename);
  }

/**************************************************************************************************
 Destructor  */

Camera::~Camera()
  {
  }

/**************************************************************************************************
 Load from file  */

bool Camera::load(char* filename)
  {
    ifstream fh;                                                    //  File handle

    fh.open(filename, ios::binary);                                 //  Open the given file (for binary reading)
    if(fh.fail())
      return false;

    fh.seekg(0, ios::beg);                                          //  Start at the beginning

    fh.read((char*)(&_header), CAMERA_HEADER_LENGTH * sizeof(char));//  Read the header

    fh.read((char*)(&_fx), sizeof(float));                          //  Read the intrinsics
    fh.read((char*)(&_s),  sizeof(float));
    fh.read((char*)(&_cx), sizeof(float));
    fh.read((char*)(&_fy), sizeof(float));
    fh.read((char*)(&_cy), sizeof(float));

    fh.read((char*)(&_k1), sizeof(float));                          //  Read the distortion coefficients
    fh.read((char*)(&_k2), sizeof(float));

    fh.read((char*)(&_p1), sizeof(float));
    fh.read((char*)(&_p2), sizeof(float));

    fh.read((char*)(&_k3), sizeof(float));
    fh.read((char*)(&_k4), sizeof(float));
    fh.read((char*)(&_k5), sizeof(float));
    fh.read((char*)(&_k6), sizeof(float));

    fh.read((char*)(&_s1), sizeof(float));
    fh.read((char*)(&_s2), sizeof(float));
    fh.read((char*)(&_s3), sizeof(float));
    fh.read((char*)(&_s4), sizeof(float));

    fh.read((char*)(&_tx), sizeof(float));
    fh.read((char*)(&_ty), sizeof(float));

    fh.close();

    return true;
  }

/**************************************************************************************************
 Setters  */

void Camera::set_fx(float x)
  {
    _fx = x;
    return;
  }

void Camera::set_fy(float x)
  {
    _fy = x;
    return;
  }

void Camera::set_shear(float x)
  {
    _s = x;
    return;
  }

void Camera::set_cx(float x)
  {
    _cx = x;
    return;
  }

void Camera::set_cy(float x)
  {
    _cy = x;
    return;
  }

void Camera::set_k1(float x)
  {
    _k1 = x;
    return;
  }

void Camera::set_k2(float x)
  {
    _k2 = x;
    return;
  }

void Camera::set_p1(float x)
  {
    _p1 = x;
    return;
  }

void Camera::set_p2(float x)
  {
    _p2 = x;
    return;
  }

void Camera::set_k3(float x)
  {
    _k3 = x;
    return;
  }

void Camera::set_k4(float x)
  {
    _k4 = x;
    return;
  }

void Camera::set_k5(float x)
  {
    _k5 = x;
    return;
  }

void Camera::set_k6(float x)
  {
    _k6 = x;
    return;
  }

void Camera::set_s1(float x)
  {
    _s1 = x;
    return;
  }

void Camera::set_s2(float x)
  {
    _s2 = x;
    return;
  }

void Camera::set_s3(float x)
  {
    _s3 = x;
    return;
  }

void Camera::set_s4(float x)
  {
    _s4 = x;
    return;
  }

void Camera::set_tx(float x)
  {
    _tx = x;
    return;
  }

void Camera::set_ty(float x)
  {
    _ty = x;
    return;
  }

/**************************************************************************************************
 Getters  */

float Camera::fx(void) const
  {
    return _fx;
  }

float Camera::fy(void) const
  {
    return _fy;
  }

float Camera::shear(void) const
  {
    return _s;
  }

float Camera::cx(void) const
  {
    return _cx;
  }

float Camera::cy(void) const
  {
    return _cy;
  }

float Camera::k1(void) const
  {
    return _k1;
  }

float Camera::k2(void) const
  {
    return _k2;
  }

float Camera::p1(void) const
  {
    return _p1;
  }

float Camera::p2(void) const
  {
    return _p2;
  }

float Camera::k3(void) const
  {
    return _k3;
  }

float Camera::k4(void) const
  {
    return _k4;
  }

float Camera::k5(void) const
  {
    return _k5;
  }

float Camera::k6(void) const
  {
    return _k6;
  }

float Camera::s1(void) const
  {
    return _s1;
  }

float Camera::s2(void) const
  {
    return _s2;
  }

float Camera::s3(void) const
  {
    return _s3;
  }

float Camera::s4(void) const
  {
    return _s4;
  }

float Camera::tx(void) const
  {
    return _tx;
  }

float Camera::ty(void) const
  {
    return _ty;
  }

/**************************************************************************************************
 Display  */

void Camera::print(void) const
  {
    cout << _header << endl;
    cout << "K:     [" << _fx << "\t" << _s  << "\t" << _cx << "]" << endl;
    cout << "       [" << 0.0 << "\t" << _fy << "\t" << _cy << "]" << endl;
    cout << "       [" << 0.0 << "\t" << 0.0 << "\t" << 1.0 << "]" << endl;

    cout << "dist:  [" << _k1 << "\t" << _k2;
    cout <<       "\t" << _p1 << "\t" << _p2;
    cout <<       "\t" << _k3 << "\t" << _k4 << "\t" << _k5 << "\t" << _k6;
    cout <<       "\t" << _s1 << "\t" << _s2 << "\t" << _s3 << "\t" << _s4;
    cout <<       "\t" << _tx << "\t" << _ty << "]" << endl;
    return;
  }

/**************************************************************************************************
 To Matrix  */

void Camera::K(cv::Mat* Kmat) const
  {
    (*Kmat) = cv::Mat(3, 3, CV_32FC1);
    (*Kmat).at<float>(0, 0) = _fx; (*Kmat).at<float>(0, 1) = _s;  (*Kmat).at<float>(0, 2) = _cx;
    (*Kmat).at<float>(1, 0) = 0.0; (*Kmat).at<float>(1, 1) = _fy; (*Kmat).at<float>(1, 2) = _cy;
    (*Kmat).at<float>(2, 0) = 0.0; (*Kmat).at<float>(2, 1) = 0.0; (*Kmat).at<float>(2, 2) = 1.0;
    return;
  }

void Camera::dist(cv::Mat* distortionCoeffs) const
  {
    (*distortionCoeffs) = cv::Mat(14, 1, CV_32FC1);
    (*distortionCoeffs).at<float>(0, 0) = _k1;
    (*distortionCoeffs).at<float>(1, 0) = _k2;
    (*distortionCoeffs).at<float>(2, 0) = _p1;
    (*distortionCoeffs).at<float>(3, 0) = _p2;
    (*distortionCoeffs).at<float>(4, 0) = _k3;
    (*distortionCoeffs).at<float>(5, 0) = _k4;
    (*distortionCoeffs).at<float>(6, 0) = _k5;
    (*distortionCoeffs).at<float>(7, 0) = _k6;
    (*distortionCoeffs).at<float>(8, 0) = _s1;
    (*distortionCoeffs).at<float>(9, 0) = _s2;
    (*distortionCoeffs).at<float>(10, 0) = _s3;
    (*distortionCoeffs).at<float>(11, 0) = _s4;
    (*distortionCoeffs).at<float>(12, 0) = _tx;
    (*distortionCoeffs).at<float>(13, 0) = _ty;
    return;
  }

/**************************************************************************************************
 Pack-up  */

/* Return the number of bytes */
unsigned int Camera::writeByteArray(char** buffer) const
  {
    unsigned int len = 0;
    unsigned int i;
    char* cast;

    #ifdef __CAMERA_DEBUG
    cout << "Camera::writeByteArray()" << endl;
    #endif

    len += CAMERA_HEADER_LENGTH;                                    //  Characters in the header

    len += (unsigned int)(sizeof(float) / sizeof(char));            //  fx
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  s
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  cx
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  fy
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  cy

    len += (unsigned int)(sizeof(float) / sizeof(char));            //  k1
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  k2
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  p1
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  p2
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  k3
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  k4
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  k5
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  k6

    len += (unsigned int)(sizeof(float) / sizeof(char));            //  s1
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  s2
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  s3
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  s4

    len += (unsigned int)(sizeof(float) / sizeof(char));            //  tx
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  ty

    if(((*buffer) = (char*)malloc(len * sizeof(char))) == NULL)
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate byte array for Camera." << endl;
        #endif
        return 0;
      }

    len = 0;                                                        //  Reset

    for(i = 0; i < CAMERA_HEADER_LENGTH; i++)                       //  Header
      (*buffer)[len + i] = _header[i];
    len += i;

    cast = (char*)(&_fx);                                           //  fx
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_s);                                            //  shear
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_cx);                                           //  cx
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_fy);                                           //  fy
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_cy);                                           //  cy
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_k1);                                           //  k1
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_k2);                                           //  k2
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_p1);                                           //  p1
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_p2);                                           //  p2
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_k3);                                           //  k3
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_k4);                                           //  k4
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_k5);                                           //  k5
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_k6);                                           //  k6
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_s1);                                           //  s1
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_s2);                                           //  s2
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_s3);                                           //  s3
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_s4);                                           //  s4
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_tx);                                           //  tx
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_ty);                                           //  ty
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    return len;
  }

/* Update attributes according to the given byte array 'buffer' */
bool Camera::readByteArray(char* buffer)
  {
    unsigned int len = 0;
    unsigned int i;
    char* cast;

    #ifdef __CAMERA_DEBUG
    cout << "Camera::readByteArray()" << endl;
    #endif

    for(i = 0; i < CAMERA_HEADER_LENGTH; i++)                       //  header
      _header[i] = buffer[len + i];
    len += i;

    //////////////////////////////////////////////////////////////////  Intrinsics

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  fx
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.fx." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_fx, cast, sizeof(_fx));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  shear
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.s." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_s, cast, sizeof(_s));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  cx
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.cx." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_cx, cast, sizeof(_cx));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  fy
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.fy." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_fy, cast, sizeof(_fy));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  cy
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.cy." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_cy, cast, sizeof(_cy));
    free(cast);
    len += i;

    //////////////////////////////////////////////////////////////////  Distortion vector

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  k1
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.k1." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_k1, cast, sizeof(_k1));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  k2
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.k2." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_k2, cast, sizeof(_k2));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  p1
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.p1." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_p1, cast, sizeof(_p1));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  p2
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.p2." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_p2, cast, sizeof(_p2));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  k3
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.k3." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_k3, cast, sizeof(_k3));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  k4
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.k4." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_k4, cast, sizeof(_k4));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  k5
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.k5." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_k5, cast, sizeof(_k5));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  k6
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.k6." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_k6, cast, sizeof(_k6));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  s1
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.s1." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_s1, cast, sizeof(_s1));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  s2
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.s2." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_s2, cast, sizeof(_s2));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  s3
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.s3." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_s3, cast, sizeof(_s3));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  s4
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.s4." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_s4, cast, sizeof(_s4));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  tx
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.tx." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_tx, cast, sizeof(_tx));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  ty
      {
        #ifdef __CAMERA_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Camera.ty." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_ty, cast, sizeof(_ty));
    free(cast);
    len += i;

    return true;
  }

#endif
