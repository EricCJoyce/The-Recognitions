#ifndef __DESCRIPTOR_CPP
#define __DESCRIPTOR_CPP

#include "descriptor.h"

/**************************************************************************************************
 (Virtual) Destructor  */

Descriptor::~Descriptor()
  {
  }

/**************************************************************************************************
 Setters  */

void Descriptor::XYZ(float x, float y, float z)
  {
    #ifdef __DESCRIPTOR_DEBUG
    cout << "Descriptor::XYZ(" << x << ", " << y << ", " << z << ")" << endl;
    #endif

    _x = x;
    _y = y;
    _z = z;
    return;
  }

void Descriptor::setSize(float s)
  {
    #ifdef __DESCRIPTOR_DEBUG
    cout << "Descriptor::setSize(" << s << ")" << endl;
    #endif

    _size = s;
    return;
  }

void Descriptor::setAngle(float ang)
  {
    #ifdef __DESCRIPTOR_DEBUG
    cout << "Descriptor::setAngle(" << ang << ")" << endl;
    #endif

    _angle = ang;
    return;
  }

void Descriptor::setResponse(float resp)
  {
    #ifdef __DESCRIPTOR_DEBUG
    cout << "Descriptor::setResponse(" << resp << ")" << endl;
    #endif

    _response = resp;
    return;
  }

void Descriptor::setOctave(signed int oct)
  {
    #ifdef __DESCRIPTOR_DEBUG
    cout << "Descriptor::setOctave(" << +oct << ")" << endl;
    #endif

    _octave = oct;
    return;
  }

void Descriptor::setSignature(signed int sig)
  {
    #ifdef __DESCRIPTOR_DEBUG
    cout << "Descriptor::setSignature(" << +sig << ")" << endl;
    #endif

    _signature = sig;
    return;
  }

/**************************************************************************************************
 Getters  */

unsigned char Descriptor::type() const
  {
    return _type;
  }

unsigned char Descriptor::len() const
  {
    return _len;
  }

float Descriptor::x() const
  {
    return _x;
  }

float Descriptor::y() const
  {
    return _y;
  }

float Descriptor::z() const
  {
    return _z;
  }

float Descriptor::size() const
  {
    return _size;
  }

float Descriptor::angle() const
  {
    return _angle;
  }

float Descriptor::response() const
  {
    return _response;
  }

signed int Descriptor::octave() const
  {
    return _octave;
  }

signed int Descriptor::signature() const
  {
    return _signature;
  }

/* Virtual function: I just want the derived classes to have a member function named vec() */
unsigned char Descriptor::vec(unsigned char** v) const
  {
    return _len;
  }

/* Virtual function: I just want the derived classes to have a member function named vec() */
unsigned char Descriptor::vec(float** v) const
  {
    return _len;
  }

/* Virtual function: I just want the derived classes to have a member function named atu() */
unsigned char Descriptor::atu(unsigned char i) const
  {
    return 0;
  }

/* Virtual function: I just want the derived classes to have a member function named atf() */
float Descriptor::atf(unsigned char i) const
  {
    return 0.0;
  }

/* Virtual function: I just want the derived classes to have a member function named print() */
void Descriptor::print(void) const
  {
    return;
  }

/**************************************************************************************************
 Derived, BRISKDesc  */

BRISKDesc::BRISKDesc()
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "BRISKDesc::BRISKDesc()" << endl;
    #endif

    _type = _BRISK;
    _len = _BRISK_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (unsigned char*)malloc(_BRISK_DESCLEN * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate BRISK descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _BRISK_DESCLEN; i++)
      _vec[i] = 0;
  }

BRISKDesc::BRISKDesc(unsigned char* v)
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "BRISKDesc::BRISKDesc(uchar*)" << endl;
    #endif

    _type = _BRISK;
    _len = _BRISK_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (unsigned char*)malloc(_BRISK_DESCLEN * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate BRISK descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _BRISK_DESCLEN; i++)
      _vec[i] = v[i];
  }

BRISKDesc::~BRISKDesc()
  {
    free(_vec);
  }

unsigned char BRISKDesc::vec(unsigned char** buffer) const
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "BRISKDesc::vec()" << endl;
    #endif

    if(((*buffer) = (unsigned char*)malloc(_BRISK_DESCLEN * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate buffer for BRISK descriptor copy." << endl;
        exit(1);
      }
    for(i = 0; i < _BRISK_DESCLEN; i++)
      (*buffer)[i] = _vec[i];

    return _BRISK_DESCLEN;
  }

/* Return the i-th element (or the last one) */
unsigned char BRISKDesc::atu(unsigned char i) const
  {
    if(i < _BRISK_DESCLEN)
      return _vec[i];
    return _vec[_BRISK_DESCLEN - 1];
  }

void BRISKDesc::print(void) const
  {
    unsigned char i;

    cout << _x << ", " << _y << ", " << _z << ": BRISK: [";
    for(i = 0; i < _BRISK_DESCLEN; i++)
      {
        cout << +_vec[i];
        if(i < _BRISK_DESCLEN - 1)
          cout << " ";
      }
    cout << "]" << endl;

    return;
  }

/**************************************************************************************************
 Derived, ORBDesc  */

ORBDesc::ORBDesc()
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "ORBDesc::ORBDesc()" << endl;
    #endif

    _type = _ORB;
    _len = _ORB_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (unsigned char*)malloc(_ORB_DESCLEN * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate ORB descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _ORB_DESCLEN; i++)
      _vec[i] = 0;
  }

ORBDesc::ORBDesc(unsigned char* v)
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "ORBDesc::ORBDesc(uchar*)" << endl;
    #endif

    _type = _ORB;
    _len = _ORB_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (unsigned char*)malloc(_ORB_DESCLEN * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate ORB descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _ORB_DESCLEN; i++)
      _vec[i] = v[i];
  }

ORBDesc::~ORBDesc()
  {
    free(_vec);
  }

unsigned char ORBDesc::vec(unsigned char** buffer) const
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "ORBDesc::vec()" << endl;
    #endif

    if(((*buffer) = (unsigned char*)malloc(_ORB_DESCLEN * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate buffer for ORB descriptor copy." << endl;
        exit(1);
      }
    for(i = 0; i < _ORB_DESCLEN; i++)
      (*buffer)[i] = _vec[i];

    return _ORB_DESCLEN;
  }

/* Return the i-th element (or the last one) */
unsigned char ORBDesc::atu(unsigned char i) const
  {
    if(i < _ORB_DESCLEN)
      return _vec[i];
    return _vec[_ORB_DESCLEN - 1];
  }

void ORBDesc::print(void) const
  {
    unsigned char i;

    cout << _x << ", " << _y << ", " << _z << ": ORB: [";
    for(i = 0; i < _ORB_DESCLEN; i++)
      {
        cout << +_vec[i];
        if(i < _ORB_DESCLEN - 1)
          cout << " ";
      }
    cout << "]" << endl;

    return;
  }

/**************************************************************************************************
 Derived, SIFTDesc  */

SIFTDesc::SIFTDesc()
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "SIFTDesc::SIFTDesc()" << endl;
    #endif

    _type = _SIFT;
    _len = _SIFT_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (float*)malloc(_SIFT_DESCLEN * sizeof(float))) == NULL)
      {
        cout << "ERROR: Unable to allocate SIFT descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _SIFT_DESCLEN; i++)
      _vec[i] = 0.0;
  }

SIFTDesc::SIFTDesc(float* v)
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "SIFTDesc::SIFTDesc(float*)" << endl;
    #endif

    _type = _SIFT;
    _len = _SIFT_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (float*)malloc(_SIFT_DESCLEN * sizeof(float))) == NULL)
      {
        cout << "ERROR: Unable to allocate SIFT descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _SIFT_DESCLEN; i++)
      _vec[i] = v[i];
  }

SIFTDesc::~SIFTDesc()
  {
    free(_vec);
  }

unsigned char SIFTDesc::vec(float** buffer) const
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "SIFTDesc::vec()" << endl;
    #endif

    if(((*buffer) = (float*)malloc(_SIFT_DESCLEN * sizeof(float))) == NULL)
      {
        cout << "ERROR: Unable to allocate buffer for SIFT descriptor copy." << endl;
        exit(1);
      }
    for(i = 0; i < _SIFT_DESCLEN; i++)
      (*buffer)[i] = _vec[i];

    return _SIFT_DESCLEN;
  }

/* Return the i-th element (or the last one) */
float SIFTDesc::atf(unsigned char i) const
  {
    if(i < _SIFT_DESCLEN)
      return _vec[i];
    return _vec[_SIFT_DESCLEN - 1];
  }

void SIFTDesc::print(void) const
  {
    unsigned char i;

    cout << _x << ", " << _y << ", " << _z << ": SIFT: [";
    for(i = 0; i < _SIFT_DESCLEN; i++)
      {
        cout << _vec[i];
        if(i < _SIFT_DESCLEN - 1)
          cout << " ";
      }
    cout << "]" << endl;

    return;
  }

/**************************************************************************************************
 Derived, SURFDesc  */

SURFDesc::SURFDesc()
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "SURFDesc::SURFDesc()" << endl;
    #endif

    _type = _SURF;
    _len = _SURF_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (float*)malloc(_SURF_DESCLEN * sizeof(float))) == NULL)
      {
        cout << "ERROR: Unable to allocate SURF descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _SURF_DESCLEN; i++)
      _vec[i] = 0.0;
  }

SURFDesc::SURFDesc(float* v)
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "SURFDesc::SURFDesc(float*)" << endl;
    #endif

    _type = _SURF;
    _len = _SURF_DESCLEN;
    _x = 0.0;
    _y = 0.0;
    _z = 0.0;
    _size = 0.0;
    _angle = 0.0;
    _response = 0.0;
    _octave = 0;
    _signature = -1;

    if((_vec = (float*)malloc(_SURF_DESCLEN * sizeof(float))) == NULL)
      {
        cout << "ERROR: Unable to allocate SURF descriptor vector." << endl;
        exit(1);
      }
    for(i = 0; i < _SURF_DESCLEN; i++)
      _vec[i] = v[i];
  }

SURFDesc::~SURFDesc()
  {
    free(_vec);
  }

unsigned char SURFDesc::vec(float** buffer) const
  {
    unsigned char i;

    #ifdef __DESCRIPTOR_DEBUG
    cout << "SURFDesc::vec()" << endl;
    #endif

    if(((*buffer) = (float*)malloc(_SURF_DESCLEN * sizeof(float))) == NULL)
      {
        cout << "ERROR: Unable to allocate buffer for SURF descriptor copy." << endl;
        exit(1);
      }
    for(i = 0; i < _SURF_DESCLEN; i++)
      (*buffer)[i] = _vec[i];

    return _SURF_DESCLEN;
  }

/* Return the i-th element (or the last one) */
float SURFDesc::atf(unsigned char i) const
  {
    if(i < _SURF_DESCLEN)
      return _vec[i];
    return _vec[_SURF_DESCLEN - 1];
  }

void SURFDesc::print(void) const
  {
    unsigned char i;

    cout << _x << ", " << _y << ", " << _z << ": SURF: [";
    for(i = 0; i < _SURF_DESCLEN; i++)
      {
        cout << _vec[i];
        if(i < _SURF_DESCLEN - 1)
          cout << " ";
      }
    cout << "]" << endl;

    return;
  }

#endif
