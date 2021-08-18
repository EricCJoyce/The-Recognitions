/*  Eric C. Joyce, Stevens Institute of Technology, 2020.

    Class for a single 3D object's representation, called that object's "signature."
    A signature may be made of one or more descriptors.
    The system is currently equipped to recognize four descriptors.
*/

#ifndef __SIGNATURE_H
#define __SIGNATURE_H

#include <fstream>
#include <opencv2/core/mat.hpp>

#include "descriptor.h"

#define SIGNATURE_HEADER_LENGTH 512

/*
#define __SIGNATURE_DEBUG 1
*/

using namespace cv;
using namespace std;

/**************************************************************************************************
 Signature  */
class Signature
  {
    public:
      Signature();                                                  //  Constructor
      ~Signature();                                                 //  Destructor

      bool load(char*);                                             //  Load a signature from file
      bool write(char*) const;                                      //  Write the signature to file using the given name
      bool toText(char*) const;                                     //  Write the signature details to a human-readable file

      unsigned int toBuffer(unsigned char, void**) const;           //  Write to a given buffer all (concatenated) feature vectors of given type
      void toMat(unsigned char, cv::Mat*) const;                    //  Write to a given matrix all features of given type

                                                                    //  Both these functions return the length of the vector in buffer:
      unsigned char desc(unsigned int, void**) const;               //  Write the i-th descriptor vector (of whatever type that is) to buffer
                                                                    //  Write the i-th descriptor vector of GIVEN type to buffer
      unsigned char descType(unsigned int, unsigned char, void**) const;

      unsigned int count(unsigned char) const;                      //  Return the count of features that use the given type of Descriptor
      unsigned char flags(void) const;                              //  Return an 8-bit "array" indicating which Descriptors are in use

      unsigned int writeByteArray(char**) const;                    //  Convert an instance of this class to a byte array
      bool readByteArray(char*);                                    //  Update an instance of this class according to given byte array

      void summary(void) const;
      void print(void) const;
      void printFilename(void) const;

      void setIndex(signed int);                                    //  Sets attribute '_index' and '_signature' of all Descriptors
      void setXYZ(unsigned int, float, float, float);               //  Sets the i-th interest point's x, y, and z components

      unsigned int numPoints() const;                               //  Getters
      unsigned int numMutex() const;
      float bboxMin(unsigned char) const;
      float bboxMax(unsigned char) const;
      unsigned char filename(char**) const;
      char* fileStem() const;                                       //  Just give me the file name without the path or the extension
      unsigned int header(char**) const;
      signed int index(void) const;
      unsigned char type(unsigned int) const;                       //  Return the #define'd type of the i-th feature
      unsigned char len(unsigned int) const;

      float x(unsigned int) const;                                  //  Return the i-th x
      float y(unsigned int) const;                                  //  Return the i-th y
      float z(unsigned int) const;                                  //  Return the i-th z
      float size(unsigned int) const;
      float angle(unsigned int) const;
      float response(unsigned int) const;
      signed int octave(unsigned int) const;

      float x(unsigned int, unsigned char) const;                   //  Return the i-th x of type
      float y(unsigned int, unsigned char) const;                   //  Return the i-th y of type
      float z(unsigned int, unsigned char) const;                   //  Return the i-th z of type
      float size(unsigned int, unsigned char) const;
      float angle(unsigned int, unsigned char) const;
      float response(unsigned int, unsigned char) const;
      signed int octave(unsigned int, unsigned char) const;

    private:
      unsigned int _numPoints;                                      //  Number of feature points in a .3df file
      unsigned int _numMutex;                                       //  Number of mutual-exclusions in a .3df file

      char* _filename;                                              //  For reference
      unsigned char filenameLen;

      unsigned int* descriptorCtr;                                  //  Number of each descriptor occurring in a .3df file

      float bboxminX, bboxminY, bboxminZ;                           //  Bounding box extrema
      float bboxmaxX, bboxmaxY, bboxmaxZ;

      char hdr[SIGNATURE_HEADER_LENGTH];                            //  File header
      signed int _index;                                            //  This Signature's index in an array of Signatures

      Descriptor** d;                                               //  Contains all descriptor vectors of all types
  };

#endif