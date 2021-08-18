/*  Eric C. Joyce, Stevens Institute of Technology, 2020.

    Class(es) for a single descriptor vector.
    A signature may be made of one or more descriptors and many such vectors.
*/

#ifndef __DESCRIPTOR_H
#define __DESCRIPTOR_H

#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _BRISK             0                                        /* Flag for BRISK */
#define _ORB               1                                        /* Flag for ORB */
#define _SIFT              2                                        /* Flag for SIFT */
#define _SURF              3                                        /* Flag for SURF */
#define _DESCRIPTOR_TOTAL  4                                        /* Total number of descriptors recognized by this system */

#define _BRISK_DESCLEN    64
#define _ORB_DESCLEN      32
#define _SIFT_DESCLEN    128
#define _SURF_DESCLEN     64

/*
#define __DESCRIPTOR_DEBUG 1
*/

using namespace std;

/**************************************************************************************************
 Base Class  */
class Descriptor
  {
    public:
      virtual ~Descriptor();                                        //  Place-holder destructor (derived classes free their vectors)
      void XYZ(float, float, float);                                //  Setters
      void setSize(float);
      void setAngle(float);
      void setResponse(float);
      void setOctave(signed int);
      void setSignature(signed int);

      unsigned char type() const;                                   //  Getters
      unsigned char len() const;
      float x() const;
      float y() const;
      float z() const;
      float size() const;
      float angle() const;
      float response() const;
      signed int octave() const;
      signed int signature() const;

      virtual unsigned char vec(unsigned char**) const;             //  Copy descriptor vector into given (uchar) buffer
      virtual unsigned char vec(float**) const;                     //  Copy descriptor vector into given (float) buffer

      virtual unsigned char atu(unsigned char) const;               //  Return i-th element from the (uchar) descriptor vector
      virtual float atf(unsigned char) const;                       //  Return i-th element from the (float) descriptor vector

      virtual void print(void) const;

    protected:
      unsigned char _type;                                          //  #defined above
      unsigned char _len;                                           //  Length of the descriptor vector
      float _x, _y, _z;
      float _size, _angle, _response;
      signed int _octave;
      signed int _signature;                                        //  Used in voting: this tracks to which object
                                                                    //  (index into 'signatures') this descriptor/feature point belongs
  };

/**************************************************************************************************
 BRISK  */
class BRISKDesc: public Descriptor
  {
    public:
      BRISKDesc();                                                  //  Default constructor
      BRISKDesc(unsigned char*);                                    //  Constructor receiving an array
      ~BRISKDesc();                                                 //  Destructor

      unsigned char vec(unsigned char**) const override;            //  Copy object's array into given buffer

      unsigned char atu(unsigned char) const override;              //  Return the i-th element

      void print(void) const override;

    private:
      unsigned char* _vec;                                          //  Array of uchars
  };

/**************************************************************************************************
 ORB  */
class ORBDesc: public Descriptor
  {
    public:
      ORBDesc();                                                    //  Default constructor
      ORBDesc(unsigned char*);                                      //  Constructor receiving an array
      ~ORBDesc();                                                   //  Destructor

      unsigned char vec(unsigned char**) const override;            //  Copy object's array into given buffer

      unsigned char atu(unsigned char) const override;              //  Return the i-th element

      void print(void) const override;

    private:
      unsigned char* _vec;                                          //  Array of uchars
  };

/**************************************************************************************************
 SIFT  */
class SIFTDesc: public Descriptor
  {
    public:
      SIFTDesc();                                                   //  Default constructor
      SIFTDesc(float*);                                             //  Constructor receiving an array
      ~SIFTDesc();                                                  //  Destructor

      unsigned char vec(float**) const override;                    //  Copy object's array into given buffer

      float atf(unsigned char) const override;                      //  Return the i-th element

      void print(void) const override;

    private:
      float* _vec;                                                  //  Array of floats
  };

/**************************************************************************************************
 SURF  */
class SURFDesc: public Descriptor
  {
    public:
      SURFDesc();                                                   //  Default constructor
      SURFDesc(float*);                                             //  Constructor receiving an array
      ~SURFDesc();                                                  //  Destructor

      unsigned char vec(float**) const override;                    //  Copy object's array into given buffer

      float atf(unsigned char) const override;                      //  Return the i-th element

      void print(void) const override;

    private:
      float* _vec;                                                  //  Array of floats
  };

#endif