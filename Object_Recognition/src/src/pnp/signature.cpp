#ifndef __SIGNATURE_CPP
#define __SIGNATURE_CPP

#include "signature.h"

/**************************************************************************************************
 Constructors  */

/* Signature constructor, no data given */
Signature::Signature()
  {
    unsigned int i;

    #ifdef __SIGNATURE_DEBUG
    cout << "Signature::Signature()" << endl;
    #endif

    _numPoints = 0;
    _numMutex = 0;
    filenameLen = 0;

    if((descriptorCtr = (unsigned int*)malloc(_DESCRIPTOR_TOTAL * sizeof(int))) == NULL)
      {
        cout << "ERROR: Unable to allocate signature descriptor-count vector." << endl;
        exit(1);
      }

    _index = -1;

    bboxminX = 0.0;
    bboxminY = 0.0;
    bboxminZ = 0.0;

    bboxmaxX = 0.0;
    bboxmaxY = 0.0;
    bboxmaxZ = 0.0;

    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)                          //  Blank out descriptor counters
      descriptorCtr[i] = 0;

    for(i = 0; i < 512; i++)                                        //  Blank out header
      hdr[i] = 0;
  }

/**************************************************************************************************
 Destructor  */

Signature::~Signature()
  {
    if(filenameLen > 0)
      free(_filename);
    free(descriptorCtr);
    if(_numPoints > 0)
      free(d);
  }

/**************************************************************************************************
 Load from file  */

/* Read the given file into a presently allocated Signature. */
bool Signature::load(char* filepath)
  {
    FILE* fp;                                                       //  .3df file

    unsigned char totalDesc;
    unsigned char flagArray;

    float x, y, z;                                                  //  Temp storage
    unsigned char desc;
    unsigned char descLen;
    float size, angle, response;
    signed int octave;

    unsigned char* ucharBuffer;
    float* floatBuffer;

    unsigned int i;
    bool good;

    #ifdef __SIGNATURE_DEBUG
    cout << "Signature::load(" << filepath << ")" << endl;
    #endif

    if(filenameLen > 0)                                             //  Did we have a file name stored already?
      free(_filename);
    if(_numPoints > 0)                                              //  Did we have Descriptors stored already?
      free(d);
    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)                          //  Reset descriptor counters
      descriptorCtr[i] = 0;

    filenameLen = strlen(filepath);                                 //  Save the length of the string
    if((_filename = (char*)malloc((filenameLen + 1) * sizeof(char))) == NULL)
      {
        cout << "ERROR: Unable to allocate signature file name buffer." << endl;
        return false;
      }
    for(i = 0; i < filenameLen; i++)                                //  Copy file name
      _filename[i] = filepath[i];
    _filename[i] = '\0';                                            //  NULL-cap the string

    if((fp = fopen(_filename, "rb")) == NULL)                       //  Open file for reading
      {
        cout << "ERROR: Unable to open \"" << _filename << "\"." << endl;
        return false;
      }

    fseek(fp, 0, SEEK_SET);                                         //  Start at the beginning

    good = ((fread(hdr, sizeof(char), 512, fp)) == 512);            //  Read the 512-byte header
    good = ((fread(&_numPoints, sizeof(int), 1, fp)) == 1);         //  unsigned int: Total number of feature-points
    good = ((fread(&_numMutex, sizeof(int), 1, fp)) == 1);          //  unsigned int: Total number of exclusivity-constraints

    good = ((fread(&totalDesc, sizeof(char), 1, fp)) == 1);         //  unsigned char: Total number of descriptors
    good = ((fread(&flagArray, sizeof(char), 1, fp)) == 1);         //  unsigned char: descriptor boolean array

    good = ((fread(&bboxminX, sizeof(float), 1, fp)) == 1);         //  Read bounding box min-min-min
    good = ((fread(&bboxminY, sizeof(float), 1, fp)) == 1);
    good = ((fread(&bboxminZ, sizeof(float), 1, fp)) == 1);

    good = ((fread(&bboxmaxX, sizeof(float), 1, fp)) == 1);         //  Read bounding box max-max-max
    good = ((fread(&bboxmaxY, sizeof(float), 1, fp)) == 1);
    good = ((fread(&bboxmaxZ, sizeof(float), 1, fp)) == 1);

    #ifdef __SIGNATURE_DEBUG
    cout << "  header: " << hdr << endl;
    cout << "  numPoints: " << +_numPoints << endl;
    cout << "  numMutex: " << +_numMutex << endl;
    cout << "  totalDesc: " << +totalDesc << endl;                  //  Total number of descriptors possible
    cout << "  flagArray: ";
    if((flagArray & 8) == 8)                                        //  SURF
      cout << "1";
    else
      cout << "0";

    if((flagArray & 4) == 4)                                        //  SIFT
      cout << "1";
    else
      cout << "0";

    if((flagArray & 2) == 2)                                        //  ORB
      cout << "1";
    else
      cout << "0";

    if((flagArray & 1) == 1)                                        // BRISK
      cout << "1";
    else
      cout << "0";
    cout << endl;

    cout << "  bbox min (" << bboxminX << ", " << bboxminY << ", " << bboxminZ <<
            "), bbox max (" << bboxmaxX << ", " << bboxmaxY << ", " << bboxmaxZ << ")" << endl;
    #endif

    if(!good)
      {
        cout << "ERROR: Unable to read from \"" << _filename << "\"." << endl;
        return false;
      }

    if((d = (Descriptor**)malloc(_numPoints * sizeof(Descriptor*))) == NULL)
      {
        cout << "ERROR: Unable to allocate descriptor-pointer buffer for \"" << _filename << "\"." << endl;
        return false;
      }

    i = 0;
    while(!feof(fp) && i < _numPoints)                              //  Continue reading feature-points and feature-vectors
      {
        good = ((fread(&desc, sizeof(char), 1, fp)) == 1);          //  Read descriptor flag

        good = ((fread(&x, sizeof(float), 1, fp)) == 1);            //  Read feature-point coordinates
        good = ((fread(&y, sizeof(float), 1, fp)) == 1);
        good = ((fread(&z, sizeof(float), 1, fp)) == 1);

        good = ((fread(&size, sizeof(float), 1, fp)) == 1);         //  Read feature-point details
        good = ((fread(&angle, sizeof(float), 1, fp)) == 1);
        good = ((fread(&response, sizeof(float), 1, fp)) == 1);
        good = ((fread(&octave, sizeof(int), 1, fp)) == 1);

        good = ((fread(&descLen, sizeof(char), 1, fp)) == 1);       //  Read feature-point vector length

        if(good)
          {
            switch(desc)
              {
                case _BRISK:
                  if((ucharBuffer = (unsigned char*)malloc(_BRISK_DESCLEN * sizeof(char))) == NULL)
                    {
                      cout << "ERROR: Unable to allocate BRISK descriptor vector buffer." << endl;
                      return false;
                    }
                                                                    //  Read in a descriptor-length's worth of chars
                  fread(ucharBuffer, sizeof(char), _BRISK_DESCLEN, fp);

                  d[i] = new BRISKDesc(ucharBuffer);                //  Create a new (BRISK) descriptor
                  d[i]->XYZ(x, y, z);                               //  Save its 3D location
                  d[i]->setSize(size);                              //  Save its details
                  d[i]->setAngle(angle);
                  d[i]->setResponse(response);
                  d[i]->setOctave(octave);

                  descriptorCtr[_BRISK]++;                          //  Increment the count for this type of descriptor

                  free(ucharBuffer);

                  i++;
                  break;

                case _ORB:
                  if((ucharBuffer = (unsigned char*)malloc(_ORB_DESCLEN * sizeof(char))) == NULL)
                    {
                      cout << "ERROR: Unable to allocate ORB descriptor vector buffer." << endl;
                      return false;
                    }
                                                                    //  Read in a descriptor-length's worth of chars
                  fread(ucharBuffer, sizeof(char), _ORB_DESCLEN, fp);

                  d[i] = new ORBDesc(ucharBuffer);                  //  Create a new (ORB) descriptor
                  d[i]->XYZ(x, y, z);                               //  Save its 3D location
                  d[i]->setSize(size);                              //  Save its details
                  d[i]->setAngle(angle);
                  d[i]->setResponse(response);
                  d[i]->setOctave(octave);

                  descriptorCtr[_ORB]++;                            //  Increment the count for this type of descriptor

                  free(ucharBuffer);

                  i++;
                  break;

                case _SIFT:
                  if((floatBuffer = (float*)malloc(_SIFT_DESCLEN * sizeof(float))) == NULL)
                    {
                      cout << "ERROR: Unable to allocate SIFT descriptor vector buffer." << endl;
                      return false;
                    }
                                                                    //  Read in a descriptor-length's worth of floats
                  fread(floatBuffer, sizeof(float), _SIFT_DESCLEN, fp);

                  d[i] = new SIFTDesc(floatBuffer);                 //  Create a new (SIFT) descriptor
                  d[i]->XYZ(x, y, z);                               //  Save its 3D location
                  d[i]->setSize(size);                              //  Save its details
                  d[i]->setAngle(angle);
                  d[i]->setResponse(response);
                  d[i]->setOctave(octave);

                  descriptorCtr[_SIFT]++;                           //  Increment the count for this type of descriptor

                  free(floatBuffer);

                  i++;
                  break;

                case _SURF:
                  if((floatBuffer = (float*)malloc(_SURF_DESCLEN * sizeof(float))) == NULL)
                    {
                      cout << "ERROR: Unable to allocate SURF descriptor vector buffer." << endl;
                      return false;
                    }
                                                                    //  Read in a descriptor-length's worth of floats
                  fread(floatBuffer, sizeof(float), _SURF_DESCLEN, fp);

                  d[i] = new SURFDesc(floatBuffer);                 //  Create a new (SURF) descriptor
                  d[i]->XYZ(x, y, z);                               //  Save its 3D location
                  d[i]->setSize(size);                              //  Save its details
                  d[i]->setAngle(angle);
                  d[i]->setResponse(response);
                  d[i]->setOctave(octave);

                  descriptorCtr[_SURF]++;                           //  Increment the count for this type of descriptor

                  free(floatBuffer);

                  i++;
                  break;
              }
          }
      }

    fclose(fp);
    return true;
  }

/**************************************************************************************************
 Write to file  */

/* Write in the .3df format */
bool Signature::write(char* fname) const
  {
    FILE* fp;                                                       //  .3df file

    unsigned char totalDesc = _DESCRIPTOR_TOTAL;
    unsigned char flagArray = 0;

    float x, y, z;                                                  //  Temp storage
    unsigned char desc;
    unsigned char descLen;
    float size, angle, response;
    signed int octave;

    unsigned char* ucharBuffer;
    float* floatBuffer;

    unsigned int i;
    bool good;

    #ifdef __SIGNATURE_DEBUG
    cout << "Signature::write(" << fname << ")" << endl;
    #endif

    if((fp = fopen(fname, "wb")) == NULL)                           //  Open file for writing
      {
        cout << "ERROR: Unable to create \"" << fname << "\"." << endl;
        return false;
      }

    good = ((fwrite(hdr, sizeof(char), 512, fp)) == 512);           //  Write the 512-byte header
    good = ((fwrite(&_numPoints, sizeof(int), 1, fp)) == 1);        //  unsigned int: Total number of feature-points
    good = ((fwrite(&_numMutex, sizeof(int), 1, fp)) == 1);         //  unsigned int: Total number of exclusivity-constraints

    if(descriptorCtr[_BRISK] > 0)
      flagArray |= 1;
    if(descriptorCtr[_ORB] > 0)
      flagArray |= 2;
    if(descriptorCtr[_SIFT] > 0)
      flagArray |= 4;
    if(descriptorCtr[_SURF] > 0)
      flagArray |= 8;

    good = ((fwrite(&totalDesc, sizeof(char), 1, fp)) == 1);        //  unsigned char: Total number of descriptors
    good = ((fwrite(&flagArray, sizeof(char), 1, fp)) == 1);        //  unsigned char: descriptor boolean array

    good = ((fwrite(&bboxminX, sizeof(float), 1, fp)) == 1);        //  Read bounding box min-min-min
    good = ((fwrite(&bboxminY, sizeof(float), 1, fp)) == 1);
    good = ((fwrite(&bboxminZ, sizeof(float), 1, fp)) == 1);

    good = ((fwrite(&bboxmaxX, sizeof(float), 1, fp)) == 1);        //  Read bounding box max-max-max
    good = ((fwrite(&bboxmaxY, sizeof(float), 1, fp)) == 1);
    good = ((fwrite(&bboxmaxZ, sizeof(float), 1, fp)) == 1);

    if(!good)
      {
        cout << "ERROR: Unable to write to \"" << fname << "\"." << endl;
        return false;
      }

    for(i = 0; i < _numPoints; i++)
      {
        desc = d[i]->type();                                        //  Get the descriptor flag
        good = ((fwrite(&desc, sizeof(char), 1, fp)) == 1);         //  Write descriptor flag

        x = d[i]->x();                                              //  Get the coordinates
        y = d[i]->y();
        z = d[i]->z();
        good = ((fwrite(&x, sizeof(float), 1, fp)) == 1);           //  Write feature-point coordinates
        good = ((fwrite(&y, sizeof(float), 1, fp)) == 1);
        good = ((fwrite(&z, sizeof(float), 1, fp)) == 1);

        size     = d[i]->size();                                    //  Get the interest point attributes
        angle    = d[i]->angle();
        response = d[i]->response();
        octave   = d[i]->octave();
        good = ((fwrite(&size, sizeof(float), 1, fp)) == 1);        //  Write feature-point details
        good = ((fwrite(&angle, sizeof(float), 1, fp)) == 1);
        good = ((fwrite(&response, sizeof(float), 1, fp)) == 1);
        good = ((fwrite(&octave, sizeof(int), 1, fp)) == 1);

        descLen = d[i]->len();
        good = ((fwrite(&descLen, sizeof(char), 1, fp)) == 1);      //  Write feature-point vector length

        if(good)
          {
            switch(desc)
              {
                case _BRISK:
                  d[i]->vec(&ucharBuffer);                          //  Copy i-th Descriptor (a BRISK) to uchar buffer
                  good = ((fwrite(ucharBuffer, sizeof(char), _BRISK_DESCLEN, fp)) == _BRISK_DESCLEN);
                  free(ucharBuffer);
                  break;

                case _ORB:
                  d[i]->vec(&ucharBuffer);                          //  Copy i-th Descriptor (an ORB) to uchar buffer
                  good = ((fwrite(ucharBuffer, sizeof(char), _ORB_DESCLEN, fp)) == _ORB_DESCLEN);
                  free(ucharBuffer);
                  break;

                case _SIFT:
                  d[i]->vec(&floatBuffer);                          //  Copy i-th Descriptor (a SIFT) to float buffer
                  good = ((fwrite(floatBuffer, sizeof(float), _SIFT_DESCLEN, fp)) == _SIFT_DESCLEN);
                  free(floatBuffer);
                  break;

                case _SURF:
                  d[i]->vec(&floatBuffer);                          //  Copy i-th Descriptor (a SURF) to float buffer
                  good = ((fwrite(floatBuffer, sizeof(float), _SURF_DESCLEN, fp)) == _SURF_DESCLEN);
                  free(floatBuffer);
                  break;
              }
          }
        if(!good)
          {
            cout << "ERROR: Write to file failed" << endl;
            return false;
          }
      }

    fclose(fp);
    return true;
  }

/* Write as ASCII text.
   Signature file details will include the original file name, though this is written to a file under the given name. */
bool Signature::toText(char* fname) const
  {
    ofstream fh;                                                    //  Output file handle
    unsigned int i, j;

    fh.open(fname);

    fh << _filename << endl;
    for(i = 0; i < filenameLen; i++)
      fh << "=";
    fh << endl << hdr << endl;

    fh << +_numPoints << " features" << endl;
    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)
      {
        fh << "  " << +descriptorCtr[i] << " ";
        switch(i)
          {
            case _BRISK:  fh << "BRISK" << endl;  break;
            case _ORB:    fh << "ORB"   << endl;  break;
            case _SIFT:   fh << "SIFT"  << endl;  break;
            case _SURF:   fh << "SURF"  << endl;  break;
          }
      }
    fh << +_numMutex << " mutual exclusions" << endl;
    fh << "Bounding box: (" << bboxminX << ", " << bboxminY << ", " << bboxminZ << ")" << endl;
    fh << "          to  (" << bboxmaxX << ", " << bboxmaxY << ", " << bboxmaxZ << ")" << endl;

    for(i = 0; i < _numPoints; i++)
      {
        fh << "(" << d[i]->x() << ", " << d[i]->y() << ", " << d[i]->z() << ") ";
        switch(d[i]->type())
          {
            case _BRISK: fh << "BRISK: ";  break;
            case _ORB:   fh << "ORB: ";    break;
            case _SIFT:  fh << "SIFT: ";   break;
            case _SURF:  fh << "SURF: ";   break;
          }

        if(d[i]->type() == _BRISK || d[i]->type() == _ORB)
          {
            for(j = 0; j < d[i]->len(); j++)                        //  Fetch and write uchars to file
              fh << d[i]->atu(j) << " ";
            fh << endl;
          }
        else
          {
            for(j = 0; j < d[i]->len(); j++)                        //  Fetch and write floats to file
              fh << d[i]->atf(j) << " ";
            fh << endl;
          }
      }

    fh.close();
    return true;
  }

/**************************************************************************************************
 Descriptor collection  */

/* Fill a buffer with concatenated vectors of the given descriptor type.

   Given 'type' in {BRISK, ORB, SIFT, SURF} write to a 1-dimensional buffer of length N x D,
   where N is the number of features in stored using descriptor 'type'
   and D is the length of that descriptor vector.
   That is, if this Signature stores features in both BRISK and SURF,
   then toBuffer(_BRISK) will fill (unsigned char*)'buffer' with N BRISK row-vectors,
   and toBuffer(_SURF) will fill (float*)'buffer' with N SURF row-vectors. */
unsigned int Signature::toBuffer(unsigned char type, void** buffer) const
  {
    unsigned int i, j;
    unsigned char k;
    unsigned int len = 0;

    unsigned char* ucharTmp;                                        //  Copy the vector from the Descriptor object
    float* floatTmp;

    #ifdef __SIGNATURE_DEBUG
    cout << "Signature::toBuffer(" << +type << ")" << endl;
    cout << "  " << +descriptorCtr[type] << " feature vectors" << endl;
    #endif

    if(descriptorCtr[type] > 0)                                     //  Are there any descriptors of the given type?
      {
        if(type == _BRISK)                                          //  Allocate the right buffer, the right amount
          {
            len = descriptorCtr[type] * _BRISK_DESCLEN;
            if(((*buffer) = malloc(len * sizeof(char))) == NULL)
              {
                #ifdef __SIGNATURE_DEBUG
                cout << "ERROR: Unable to allocate unsigned char buffer for BRISK output to void*." << endl;
                #endif
                return 0;
              }
          }
        else if(type == _ORB)
          {
            len = descriptorCtr[type] * _ORB_DESCLEN;
            if(((*buffer) = malloc(len * sizeof(char))) == NULL)
              {
                #ifdef __SIGNATURE_DEBUG
                cout << "ERROR: Unable to allocate unsigned char buffer for ORB output to void*." << endl;
                #endif
                return 0;
              }
          }
        else if(type == _SIFT)
          {
            len = descriptorCtr[type] * _SIFT_DESCLEN;
            if(((*buffer) = malloc(len * sizeof(float))) == NULL)
              {
                #ifdef __SIGNATURE_DEBUG
                cout << "ERROR: Unable to allocate float buffer for SIFT output to void*." << endl;
                #endif
                return 0;
              }
          }
        else if(type == _SURF)
          {
            len = descriptorCtr[type] * _SURF_DESCLEN;
            if(((*buffer) = malloc(len * sizeof(float))) == NULL)
              {
                #ifdef __SIGNATURE_DEBUG
                cout << "ERROR: Unable to allocate float buffer for SURF output to void*." << endl;
                #endif
                return 0;
              }
          }

        j = 0;                                                      //  Offset into the buffer
        for(i = 0; i < _numPoints; i++)                             //  Now that we've allocated space, fill in that space
          {
            if(d[i]->type() == type)                                //  If this is a Descriptor of the type we asked for...
              {
                if(type == _BRISK)
                  {
                    d[i]->vec(&ucharTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _BRISK_DESCLEN; k++)             //  Copy the temp into the total buffer
                      (*((unsigned char**)buffer))[j * _BRISK_DESCLEN + k] = ucharTmp[k];
                    free(ucharTmp);                                 //  Dump temp
                  }
                else if(type == _ORB)
                  {
                    d[i]->vec(&ucharTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _ORB_DESCLEN; k++)               //  Copy the temp into the total buffer
                      (*((unsigned char**)buffer))[j * _ORB_DESCLEN + k] = ucharTmp[k];
                    free(ucharTmp);                                 //  Dump temp
                  }
                else if(type == _SIFT)
                  {
                    d[i]->vec(&floatTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _SIFT_DESCLEN; k++)              //  Copy the temp into the total buffer
                      (*((float**)buffer))[j * _SIFT_DESCLEN + k] = floatTmp[k];
                    free(floatTmp);                                 //  Dump temp
                  }
                else if(type == _SURF)
                  {
                    d[i]->vec(&floatTmp);                           //  Copy this descriptor to temp
                    for(k = 0; k < _SURF_DESCLEN; k++)              //  Copy the temp into the total buffer
                      (*((float**)buffer))[j * _SURF_DESCLEN + k] = floatTmp[k];
                    free(floatTmp);                                 //  Dump temp
                  }

                j++;                                                //  Increment offset into buffer
              }
          }
      }

    return len;
  }

/* Export a cv::Mat using a specific descriptor type.

   Compare to Extractor::descMat().

   Given 'type' in {BRISK, ORB, SIFT, SURF} return an N x D cv::Mat
   where N is the number of features in stored using descriptor 'type'
   and D is the length of that descriptor vector.
   That is, if this Signature stores features in both BRISK and SURF,
   then toMat(_BRISK, mat) will fill 'mat' with N BRISK row-vectors,
   and descMat(_SURF, mat) will fill 'mat' with N SURF row-vectors. */
void Signature::toMat(unsigned char type, cv::Mat* mat) const
  {
    unsigned char* ucharBuffer;                                     //  Accumulators
    float* floatBuffer;

    #ifdef __SIGNATURE_DEBUG
    cout << "Signature::toMat(" << +type << ")" << endl;
    cout << "  " << +descriptorCtr[type] << " feature vectors" << endl;
    #endif

    if(descriptorCtr[type] > 0)                                     //  Are there any descriptors of the given type?
      {
        if(type == _BRISK)
          {
            toBuffer(_BRISK, (void**)(&ucharBuffer));
            (*mat) = cv::Mat(descriptorCtr[_BRISK], _BRISK_DESCLEN, CV_8U);
            memcpy(mat->data, ucharBuffer, descriptorCtr[_BRISK] * _BRISK_DESCLEN * sizeof(char));
            free(ucharBuffer);
          }
        else if(type == _ORB)
          {
            toBuffer(_ORB, (void**)(&ucharBuffer));
            (*mat) = cv::Mat(descriptorCtr[_ORB], _ORB_DESCLEN, CV_8U);
            memcpy(mat->data, ucharBuffer, descriptorCtr[_ORB] * _ORB_DESCLEN * sizeof(char));
            free(ucharBuffer);
          }
        else if(type == _SIFT)
          {
            toBuffer(_SIFT, (void**)(&floatBuffer));
            (*mat) = cv::Mat(descriptorCtr[_SIFT], _SIFT_DESCLEN, CV_32F);
            memcpy(mat->data, floatBuffer, descriptorCtr[_SIFT] * _SIFT_DESCLEN * sizeof(float));
            free(floatBuffer);
          }
        else if(type == _SURF)
          {
            toBuffer(_SURF, (void**)(&floatBuffer));
            (*mat) = cv::Mat(descriptorCtr[_SURF], _SURF_DESCLEN, CV_32F, floatBuffer);
            memcpy(mat->data, floatBuffer, descriptorCtr[_SURF] * _SURF_DESCLEN * sizeof(float));
            free(floatBuffer);
          }
      }

    return;
  }

/* Write the i-th descriptor vector (of whatever type that is) to buffer */
unsigned char Signature::desc(unsigned int i, void** buffer) const
  {
    unsigned char len = 0;
    unsigned char j;

    if(i < _numPoints)
      {
        switch(d[i]->type())
          {
            case _BRISK:  len = d[i]->len();
                          if(((*buffer) = malloc(len * sizeof(char))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate void* buffer for " << +i << "-th feature." << endl;
                              #endif
                              return 0;
                            }
                          for(j = 0; j < len; j++)
                            (*((unsigned char**)buffer))[j] = d[i]->atu(j);
                          break;

            case _ORB:    len = d[i]->len();
                          if(((*buffer) = malloc(len * sizeof(char))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate void* buffer for " << +i << "-th feature." << endl;
                              #endif
                              return 0;
                            }
                          for(j = 0; j < len; j++)
                            (*((unsigned char**)buffer))[j] = d[i]->atu(j);
                          break;

            case _SIFT:   len = d[i]->len();
                          if(((*buffer) = malloc(len * sizeof(float))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate void* buffer for " << +i << "-th feature." << endl;
                              #endif
                              return 0;
                            }
                          for(j = 0; j < len; j++)
                            (*((float**)buffer))[j] = d[i]->atf(j);
                          break;

            case _SURF:   len = d[i]->len();
                          if(((*buffer) = malloc(len * sizeof(float))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate void* buffer for " << +i << "-th feature." << endl;
                              #endif
                              return 0;
                            }
                          for(j = 0; j < len; j++)
                            (*((float**)buffer))[j] = d[i]->atf(j);
                          break;
          }
      }

    return len;
  }

/* Write the i-th descriptor vector of GIVEN type to buffer */
unsigned char Signature::descType(unsigned int i, unsigned char type, void** buffer) const
  {
    unsigned int index;
    unsigned int j;

    if(descriptorCtr[type] > 0)                                     //  Are there ANY Descriptors of this type?
      {
        index = 0;
        j = 0;
        while(index < _numPoints && j != i)
          {
            if(d[index]->type() == type)
              j++;

            index++;
          }
        return desc(j, buffer);
      }

    return 0;
  }

/**************************************************************************************************
 Details  */

/* Return the count of features that use the given type of Descriptor */
unsigned int Signature::count(unsigned char type) const
  {
    if(type < _DESCRIPTOR_TOTAL)
      return descriptorCtr[type];
    return 0;
  }

/* Return a flag-array (uchar) indicating which descriptors are in this Signature */
unsigned char Signature::flags(void) const
  {
    unsigned char f = 0;

    if(descriptorCtr[_BRISK] > 0)
      f |= 1;
    if(descriptorCtr[_ORB] > 0)
      f |= 2;
    if(descriptorCtr[_SIFT] > 0)
      f |= 4;
    if(descriptorCtr[_SURF] > 0)
      f |= 8;

    return f;
  }

/**************************************************************************************************
 Pack-up  */

/* Return the number of bytes */
unsigned int Signature::writeByteArray(char** buffer) const
  {
    unsigned int len = 0;
    char* cast;
    unsigned int i, j;
    unsigned char k;

    float tmpFloat;
    int tmpInt;

    unsigned char* ucharBuffer;
    float* floatBuffer;

    #ifdef __SIGNATURE_DEBUG
    cout << "Signature::writeByteArray()" << endl;
    #endif

    len += SIGNATURE_HEADER_LENGTH;                                 //  Characters in the header
    len += (unsigned int)(sizeof(int) / sizeof(char));              //  numPoints
    len += (unsigned int)(sizeof(int) / sizeof(char));              //  numMutex
    len += filenameLen + 1;                                         //  filenameLen characters, plus the char for filenameLen
                                                                    //  A count for each descriptor the system can handle
    len += (unsigned int)(sizeof(int) / sizeof(char)) * _DESCRIPTOR_TOTAL;

    len += (unsigned int)(sizeof(float) / sizeof(char));            //  bboxminX
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  bboxminY
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  bboxminZ

    len += (unsigned int)(sizeof(float) / sizeof(char));            //  bboxmaxX
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  bboxmaxY
    len += (unsigned int)(sizeof(float) / sizeof(char));            //  bboxmaxZ

    len += (unsigned int)(sizeof(int) / sizeof(char));              //  index

    for(i = 0; i < _numPoints; i++)
      {
        len += 2;                                                   //  type and len

        len += (unsigned int)(sizeof(float) / sizeof(char));        //  x
        len += (unsigned int)(sizeof(float) / sizeof(char));        //  y
        len += (unsigned int)(sizeof(float) / sizeof(char));        //  z

        len += (unsigned int)(sizeof(float) / sizeof(char));        //  size
        len += (unsigned int)(sizeof(float) / sizeof(char));        //  angle
        len += (unsigned int)(sizeof(float) / sizeof(char));        //  response

        len += (unsigned int)(sizeof(int) / sizeof(char));          //  octave

        len += (unsigned int)(sizeof(int) / sizeof(char));          //  signature

        switch(d[i]->type())                                        //  Add the descriptor lengths
          {
            case _BRISK:  len += _BRISK_DESCLEN;  break;            //  Characters
            case _ORB:    len += _ORB_DESCLEN;    break;            //  Characters
                                                                    //  Floats
            case _SIFT:   len += _SIFT_DESCLEN * (unsigned int)(sizeof(float) / sizeof(char));
                                                  break;
                                                                    //  Floats
            case _SURF:   len += _SURF_DESCLEN * (unsigned int)(sizeof(float) / sizeof(char));
                                                  break;
          }
      }

    if(((*buffer) = (char*)malloc(len * sizeof(char))) == NULL)
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate byte array for Signature." << endl;
        #endif
        return 0;
      }

    len = 0;                                                        //  Reset

    for(i = 0; i < SIGNATURE_HEADER_LENGTH; i++)                    //  Header
      (*buffer)[len + i] = hdr[i];
    len += i;

    cast = (char*)(&_numPoints);                                    //  numPoints
    for(i = 0; i < (unsigned char)(sizeof(int) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_numMutex);                                     //  numMutex
    for(i = 0; i < (unsigned char)(sizeof(int) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    (*buffer)[len] = filenameLen;                                   //  filenameLen
    len++;

    for(i = 0; i < filenameLen; i++)                                //  filename
      (*buffer)[len + i] = _filename[i];
    len += filenameLen;

    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)                          //  Counts for each descriptor type
      {
        tmpInt = descriptorCtr[i];
        cast = (char*)(&tmpInt);                                    //  descriptorCtr[i]
        for(j = 0; j < (unsigned char)(sizeof(int) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;
      }

    cast = (char*)(&bboxminX);                                      //  bboxminX
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&bboxminY);                                      //  bboxminY
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&bboxminZ);                                      //  bboxminZ
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&bboxmaxX);                                      //  bboxmaxX
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&bboxmaxY);                                      //  bboxmaxY
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&bboxmaxZ);                                      //  bboxmaxZ
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    cast = (char*)(&_index);                                        //  index
    for(i = 0; i < (unsigned char)(sizeof(int) / sizeof(char)); i++)
      (*buffer)[len + i] = cast[i];
    len += i;

    for(i = 0; i < _numPoints; i++)
      {
        (*buffer)[len] = d[i]->type();
        len++;

        switch(d[i]->type())
          {
            case _BRISK:  (*buffer)[len] = _BRISK_DESCLEN;  break;
            case _ORB:    (*buffer)[len] = _ORB_DESCLEN;    break;
            case _SIFT:   (*buffer)[len] = _SIFT_DESCLEN;   break;
            case _SURF:   (*buffer)[len] = _SURF_DESCLEN;   break;
          }
        len++;

        tmpFloat = d[i]->x();                                       //  x
        cast = (char*)(&tmpFloat);
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        tmpFloat = d[i]->y();                                       //  y
        cast = (char*)(&tmpFloat);
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        tmpFloat = d[i]->z();                                       //  z
        cast = (char*)(&tmpFloat);
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        tmpFloat = d[i]->size();                                    //  size
        cast = (char*)(&tmpFloat);
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        tmpFloat = d[i]->angle();                                   //  angle
        cast = (char*)(&tmpFloat);
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        tmpFloat = d[i]->response();                                //  response
        cast = (char*)(&tmpFloat);
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        tmpInt = d[i]->octave();                                    //  octave
        cast = (char*)(&tmpInt);
        for(j = 0; j < (unsigned char)(sizeof(int) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        tmpInt = d[i]->signature();                                 //  signature
        cast = (char*)(&tmpInt);
        for(j = 0; j < (unsigned char)(sizeof(int) / sizeof(char)); j++)
          (*buffer)[len + j] = cast[j];
        len += j;

        switch(d[i]->type())
          {
            case _BRISK:  d[i]->vec(&ucharBuffer);
                          for(j = 0; j < _BRISK_DESCLEN; j++)
                            (*buffer)[len + j] = ucharBuffer[j];
                          len += j;
                          free(ucharBuffer);
                          break;

            case _ORB:    d[i]->vec(&ucharBuffer);
                          for(j = 0; j < _ORB_DESCLEN; j++)
                            (*buffer)[len + j] = ucharBuffer[j];
                          len += j;
                          free(ucharBuffer);
                          break;

            case _SIFT:   d[i]->vec(&floatBuffer);
                          for(j = 0; j < _SIFT_DESCLEN; j++)
                            {
                              cast = (char*)(&floatBuffer[j]);
                              for(k = 0; k < (unsigned char)(sizeof(float) / sizeof(char)); k++)
                                (*buffer)[len + k] = cast[k];
                              len += k;
                            }
                          free(floatBuffer);
                          break;

            case _SURF:   d[i]->vec(&floatBuffer);
                          for(j = 0; j < _SURF_DESCLEN; j++)
                            {
                              cast = (char*)(&floatBuffer[j]);
                              for(k = 0; k < (unsigned char)(sizeof(float) / sizeof(char)); k++)
                                (*buffer)[len + k] = cast[k];
                              len += k;
                            }
                          free(floatBuffer);
                          break;
          }
      }

    return len;
  }

/* Update attributes according to the given byte array 'buffer' */
bool Signature::readByteArray(char* buffer)
  {
    unsigned int len = 0;
    unsigned int i, j;
    unsigned char k;
    char* cast;

    unsigned char tmp_type;
    float tmp_x, tmp_y, tmp_z;
    float tmp_size, tmp_angle, tmp_response;
    int tmp_octave, tmp_signature;
    unsigned char* ucharBuffer;
    float* floatBuffer;

    #ifdef __SIGNATURE_DEBUG
    cout << "Signature::readByteArray()" << endl;
    #endif

    if(filenameLen > 0)                                             //  Was there an existing file name?
      free(_filename);
    free(descriptorCtr);
    if(_numPoints > 0)                                              //  Were there any existing descriptors?
      free(d);

    for(i = 0; i < SIGNATURE_HEADER_LENGTH; i++)                    //  header
      hdr[i] = buffer[len + i];
    len += i;

    if((cast = (char*)malloc(sizeof(int))) == NULL)                 //  numPoints
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.numPoints." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(int) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_numPoints, cast, sizeof(_numPoints));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(int))) == NULL)                 //  numMutex
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.numMutex." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(int) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_numMutex, cast, sizeof(_numMutex));
    free(cast);
    len += i;

    filenameLen = buffer[len];                                      //  filenameLen
    len++;

    if((_filename = (char*)malloc((filenameLen + 1) * sizeof(char))) == NULL)
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate Signature.filename." << endl;
        #endif
        return false;
      }
    for(i = 0; i < filenameLen; i++)                                //  filename
      _filename[i] = buffer[len + i];
    _filename[i] = '\0';
    len += i;

    if((descriptorCtr = (unsigned int*)malloc(_DESCRIPTOR_TOTAL * sizeof(int))) == NULL)
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate Signature.descriptorCtr." << endl;
        #endif
        return false;
      }

    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)                          //  Load up descriptorCtr
      {
        if((cast = (char*)malloc(sizeof(int))) == NULL)             //  descriptorCtr[i]
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature.descriptorCtr[" << +i << "]." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(int) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(descriptorCtr + i, cast, sizeof(int));
        free(cast);
        len += j;
      }

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  bboxminX
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.bboxminX." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&bboxminX, cast, sizeof(bboxminX));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  bboxminY
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.bboxminY." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&bboxminY, cast, sizeof(bboxminY));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  bboxminZ
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.bboxminZ." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&bboxminZ, cast, sizeof(bboxminZ));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  bboxmaxX
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.bboxmaxX." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&bboxmaxX, cast, sizeof(bboxmaxX));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  bboxmaxY
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.bboxmaxY." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&bboxmaxY, cast, sizeof(bboxmaxY));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(float))) == NULL)               //  bboxmaxZ
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.bboxmaxZ." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(float) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&bboxmaxZ, cast, sizeof(bboxmaxZ));
    free(cast);
    len += i;

    if((cast = (char*)malloc(sizeof(int))) == NULL)                 //  index
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate temporary byte array for reading Signature.index." << endl;
        #endif
        return false;
      }
    for(i = 0; i < (unsigned char)(sizeof(int) / sizeof(char)); i++)
      cast[i] = buffer[len + i];
    memcpy(&_index, cast, sizeof(_index));
    free(cast);
    len += i;

    if((d = (Descriptor**)malloc(_numPoints * sizeof(Descriptor*))) == NULL)
      {
        #ifdef __SIGNATURE_DEBUG
        cout << "ERROR: Unable to allocate descriptor-pointer buffer from byte-array." << endl;
        #endif
        return false;
      }

    for(i = 0; i < _numPoints; i++)
      {
        tmp_type = buffer[len];                                     //  Skip the length; we'll look it up
        len += 2;

        if((cast = (char*)malloc(sizeof(float))) == NULL)           //  x
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].x." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_x, cast, sizeof(tmp_x));
        free(cast);
        len += j;

        if((cast = (char*)malloc(sizeof(float))) == NULL)           //  y
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].y." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_y, cast, sizeof(tmp_y));
        free(cast);
        len += j;

        if((cast = (char*)malloc(sizeof(float))) == NULL)           //  z
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].z." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_z, cast, sizeof(tmp_z));
        free(cast);
        len += j;

        if((cast = (char*)malloc(sizeof(float))) == NULL)           //  size
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].size." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_size, cast, sizeof(tmp_size));
        free(cast);
        len += j;

        if((cast = (char*)malloc(sizeof(float))) == NULL)           //  angle
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].angle." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_angle, cast, sizeof(tmp_angle));
        free(cast);
        len += j;

        if((cast = (char*)malloc(sizeof(float))) == NULL)           //  response
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].response." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_response, cast, sizeof(tmp_response));
        free(cast);
        len += j;

        if((cast = (char*)malloc(sizeof(float))) == NULL)           //  octave
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].octave." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(float) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_octave, cast, sizeof(tmp_octave));
        free(cast);
        len += j;

        if((cast = (char*)malloc(sizeof(int))) == NULL)             //  signature
          {
            #ifdef __SIGNATURE_DEBUG
            cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].signature." << endl;
            #endif
            return false;
          }
        for(j = 0; j < (unsigned char)(sizeof(int) / sizeof(char)); j++)
          cast[j] = buffer[len + j];
        memcpy(&tmp_signature, cast, sizeof(tmp_signature));
        free(cast);
        len += j;

        switch(tmp_type)
          {
            case _BRISK:  if((ucharBuffer = (unsigned char*)malloc(_BRISK_DESCLEN * sizeof(char))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate temporary uchar array for reading Signature[" << +i << "] vector." << endl;
                              #endif
                              return false;
                            }
                          for(j = 0; j < _BRISK_DESCLEN; j++)
                            ucharBuffer[j] = buffer[len + j];
                          len += j;
                          d[i] = new BRISKDesc(ucharBuffer);
                          d[i]->XYZ(tmp_x, tmp_y, tmp_z);
                          d[i]->setSize(tmp_size);
                          d[i]->setAngle(tmp_angle);
                          d[i]->setResponse(tmp_response);
                          d[i]->setOctave(tmp_octave);
                          d[i]->setSignature(tmp_signature);
                          free(ucharBuffer);
                          break;

            case _ORB:    if((ucharBuffer = (unsigned char*)malloc(_ORB_DESCLEN * sizeof(char))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate temporary uchar array for reading Signature[" << +i << "] vector." << endl;
                              #endif
                              return false;
                            }
                          for(j = 0; j < _ORB_DESCLEN; j++)
                            ucharBuffer[j] = buffer[len + j];
                          len += j;
                          d[i] = new ORBDesc(ucharBuffer);
                          d[i]->XYZ(tmp_x, tmp_y, tmp_z);
                          d[i]->setSize(tmp_size);
                          d[i]->setAngle(tmp_angle);
                          d[i]->setResponse(tmp_response);
                          d[i]->setOctave(tmp_octave);
                          d[i]->setSignature(tmp_signature);
                          free(ucharBuffer);
                          break;

            case _SIFT:   if((floatBuffer = (float*)malloc(_SIFT_DESCLEN * sizeof(float))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate temporary float array for reading Signature[" << +i << "] vector." << endl;
                              #endif
                              return false;
                            }
                          if((cast = (char*)malloc(sizeof(float))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].response." << endl;
                              #endif
                              return false;
                            }

                          for(j = 0; j < _SIFT_DESCLEN; j++)
                            {
                              for(k = 0; k < (unsigned char)(sizeof(float) / sizeof(char)); k++)
                                cast[k] = buffer[len + k];
                              memcpy(&floatBuffer[j], cast, sizeof(float));
                              len += k;
                            }

                          free(cast);

                          d[i] = new SIFTDesc(floatBuffer);
                          d[i]->XYZ(tmp_x, tmp_y, tmp_z);
                          d[i]->setSize(tmp_size);
                          d[i]->setAngle(tmp_angle);
                          d[i]->setResponse(tmp_response);
                          d[i]->setOctave(tmp_octave);
                          d[i]->setSignature(tmp_signature);
                          free(floatBuffer);
                          break;

            case _SURF:   if((floatBuffer = (float*)malloc(_SURF_DESCLEN * sizeof(float))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate temporary float array for reading Signature[" << +i << "] vector." << endl;
                              #endif
                              return false;
                            }
                          if((cast = (char*)malloc(sizeof(float))) == NULL)
                            {
                              #ifdef __SIGNATURE_DEBUG
                              cout << "ERROR: Unable to allocate temporary byte array for reading Signature[" << +i << "].response." << endl;
                              #endif
                              return false;
                            }

                          for(j = 0; j < _SURF_DESCLEN; j++)
                            {
                              for(k = 0; k < (unsigned char)(sizeof(float) / sizeof(char)); k++)
                                cast[k] = buffer[len + k];
                              memcpy(&floatBuffer[j], cast, sizeof(float));
                              len += k;
                            }

                          free(cast);

                          d[i] = new SURFDesc(floatBuffer);
                          d[i]->XYZ(tmp_x, tmp_y, tmp_z);
                          d[i]->setSize(tmp_size);
                          d[i]->setAngle(tmp_angle);
                          d[i]->setResponse(tmp_response);
                          d[i]->setOctave(tmp_octave);
                          d[i]->setSignature(tmp_signature);
                          free(floatBuffer);
                          break;
          }
      }

    return true;
  }

/**************************************************************************************************
 Display  */

void Signature::summary(void) const
  {
    unsigned int i;

    cout << _filename << endl;
    for(i = 0; i < filenameLen; i++)
      cout << "=";
    cout << endl << hdr << endl;

    cout << +_numPoints << " features" << endl;
    for(i = 0; i < _DESCRIPTOR_TOTAL; i++)
      {
        cout << "  " << +descriptorCtr[i] << " ";
        switch(i)
          {
            case _BRISK:  cout << "BRISK" << endl;  break;
            case _ORB:    cout << "ORB" << endl;    break;
            case _SIFT:   cout << "SIFT" << endl;   break;
            case _SURF:   cout << "SURF" << endl;   break;
          }
      }
    cout << +_numMutex << " mutual exclusions" << endl;
    cout << "Bounding box: (" << bboxminX << ", " << bboxminY << ", " << bboxminZ << ")" << endl;
    cout << "          to  (" << bboxmaxX << ", " << bboxmaxY << ", " << bboxmaxZ << ")" << endl;

    return;
  }

void Signature::print(void) const
  {
    unsigned int i;

    summary();

    for(i = 0; i < _numPoints; i++)
      d[i]->print();

    return;
  }

void Signature::printFilename(void) const
  {
    cout << _filename;
    return;
  }

/**************************************************************************************************
 Setters  */

/* Sets attribute '_index' and '_signature' of all Descriptors */
void Signature::setIndex(signed int index)
  {
    unsigned int i;

    _index = index;                                                 //  Set the Signature's '_index'
    for(i = 0; i < _numPoints; i++)                                 //  Set all Descriptors' '_signature's
      d[i]->setSignature(index);

    return;
  }

void Signature::setXYZ(unsigned int i, float x, float y, float z)
  {
    if(i < _numPoints)
      d[i]->XYZ(x, y, z);
    return;
  }

/**************************************************************************************************
 Getters  */

/* Return the number of interest points in this signature */
unsigned int Signature::numPoints() const
  {
    return _numPoints;
  }

/* Return the number of mutual exclusions in this signature */
unsigned int Signature::numMutex() const
  {
    return _numMutex;
  }

/* Return the coord-th coordinate of this signature's minimum bounding box vertex.
   This method can receive characters (upper and lower case) and indices in {0, 1, 2}. */
float Signature::bboxMin(unsigned char coord) const
  {
    switch(coord)
      {
        case 0:
        case 'x':
        case 'X': return bboxminX;

        case 1:
        case 'y':
        case 'Y': return bboxminY;

        case 2:
        case 'z':
        case 'Z': return bboxminZ;
      }

    return -INFINITY;
  }

/* Return the coord-th coordinate of this signature's maximum bounding box vertex.
   This method can receive characters (upper and lower case) and indices in {0, 1, 2}. */
float Signature::bboxMax(unsigned char coord) const
  {
    switch(coord)
      {
        case 0:
        case 'x':
        case 'X': return bboxmaxX;

        case 1:
        case 'y':
        case 'Y': return bboxmaxY;

        case 2:
        case 'z':
        case 'Z': return bboxmaxZ;
      }

    return INFINITY;
  }

/* Copy this signature's file name to the given buffer and return the strings's length. */
unsigned char Signature::filename(char** buffer) const
  {
    unsigned char i;
    if(((*buffer) = (char*)malloc((filenameLen + 1) * sizeof(char))) == NULL)
      return 0;
    for(i = 0; i < filenameLen; i++)
      (*buffer)[i] = _filename[i];
    (*buffer)[i] = '\0';
    return filenameLen;
  }

/* Build and return a string that is just the file name without the path or the extension. */
char* Signature::fileStem() const
  {
    char* ret;
    signed int i = (signed int)filenameLen - 1;
    signed int j;
    unsigned char k;

    while(i > 0 && _filename[i] != '.')                             //  Find the '.' before the file extension
      i--;
    j = i;
    while(j > 0 && _filename[j] != '/')                             //  Find the '/' before the file path (or the head of the string)
      j--;

    if((ret = (char*)malloc((i - j) * sizeof(char))) == NULL)
      return NULL;

    for(k = 0; k < (i - j - 1); k++)
      ret[k] = _filename[j + 1 + k];
    ret[k] = '\0';                                                  //  NULL-cap

    return ret;
  }

/* Copy this signature's header to the given buffer and return the header's length
   (which is always equal to SIGNATURE_HEADER_LENGTH). */
unsigned int Signature::header(char** buffer) const
  {
    unsigned int i = 0;
    if(((*buffer) = (char*)malloc(SIGNATURE_HEADER_LENGTH * sizeof(char))) == NULL)
      return 0;
    while(i < SIGNATURE_HEADER_LENGTH && hdr[i] != '\0')
      {
        (*buffer)[i] = hdr[i];
        i++;
      }
    for(; i < SIGNATURE_HEADER_LENGTH; i++)
      (*buffer)[i] = '\0';
    return SIGNATURE_HEADER_LENGTH;
  }

/* A signature's 'index' should be equal to its position in the array of all signatures in the partent program.
   Unless this value is set, it is -1. */
signed int Signature::index(void) const
  {
    return _index;
  }

/* Return {_BRISK, _ORB, _SIFT, _SURF} according to the type of the i-th Descriptor */
unsigned char Signature::type(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->type();
    return _DESCRIPTOR_TOTAL;
  }

/* Return the length of the i-th feature descriptor vector. */
unsigned char Signature::len(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->len();
    return 0;
  }

/* Return the i-th interest point's X coordinate */
float Signature::x(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->x();
    return INFINITY;
  }

/* Return the i-th interest point's Y coordinate */
float Signature::y(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->y();
    return INFINITY;
  }

/* Return the i-th interest point's Z coordinate */
float Signature::z(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->z();
    return INFINITY;
  }

/* Return the i-th interest point's size */
float Signature::size(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->size();
    return INFINITY;
  }

/* Return the i-th interest point's angle */
float Signature::angle(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->angle();
    return INFINITY;
  }

/* Return the i-th interest point's response */
float Signature::response(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->response();
    return INFINITY;
  }

/* Return the i-th interest point's octave */
signed int Signature::octave(unsigned int i) const
  {
    if(i < _numPoints)
      return d[i]->octave();
    return INT_MAX;
  }

/* Return the X-coordinate of the i-th (or last) interest point that is described using 'type'.
   If there are no interest points that use this type, return infinity. */
float Signature::x(unsigned int index, unsigned char type) const
  {
    unsigned int i, j;
    if(descriptorCtr[type] > 0)
      {
        i = 0;
        j = 0;
        while(i < _numPoints && !(j == index && d[j]->type() == type))
          {
            if(d[j]->type() == type)
              j++;
            i++;
          }
        return d[j]->x();
      }
    return INFINITY;
  }

/* Return the Y-coordinate of the i-th (or last) interest point that is described using 'type'.
   If there are no interest points that use this type, return infinity. */
float Signature::y(unsigned int index, unsigned char type) const
  {
    unsigned int i, j;
    if(descriptorCtr[type] > 0)
      {
        i = 0;
        j = 0;
        while(i < _numPoints && !(j == index && d[j]->type() == type))
          {
            if(d[j]->type() == type)
              j++;
            i++;
          }
        return d[j]->y();
      }
    return INFINITY;
  }

/* Return the Z-coordinate of the i-th (or last) interest point that is described using 'type'.
   If there are no interest points that use this type, return infinity. */
float Signature::z(unsigned int index, unsigned char type) const
  {
    unsigned int i, j;
    if(descriptorCtr[type] > 0)
      {
        i = 0;
        j = 0;
        while(i < _numPoints && !(j == index && d[j]->type() == type))
          {
            if(d[j]->type() == type)
              j++;
            i++;
          }
        return d[j]->z();
      }
    return INFINITY;
  }

/* Return the size of the i-th (or last) interest point that is described using 'type'.
   If there are no interest points that use this type, return infinity. */
float Signature::size(unsigned int index, unsigned char type) const
  {
    unsigned int i, j;
    if(descriptorCtr[type] > 0)
      {
        i = 0;
        j = 0;
        while(i < _numPoints && !(j == index && d[j]->type() == type))
          {
            if(d[j]->type() == type)
              j++;
            i++;
          }
        return d[j]->size();
      }
    return INFINITY;
  }

/* Return the angle of the i-th (or last) interest point that is described using 'type'.
   If there are no interest points that use this type, return infinity. */
float Signature::angle(unsigned int index, unsigned char type) const
  {
    unsigned int i, j;
    if(descriptorCtr[type] > 0)
      {
        i = 0;
        j = 0;
        while(i < _numPoints && !(j == index && d[j]->type() == type))
          {
            if(d[j]->type() == type)
              j++;
            i++;
          }
        return d[j]->angle();
      }
    return INFINITY;
  }

/* Return the response of the i-th (or last) interest point that is described using 'type'.
   If there are no interest points that use this type, return infinity. */
float Signature::response(unsigned int index, unsigned char type) const
  {
    unsigned int i, j;
    if(descriptorCtr[type] > 0)
      {
        i = 0;
        j = 0;
        while(i < _numPoints && !(j == index && d[j]->type() == type))
          {
            if(d[j]->type() == type)
              j++;
            i++;
          }
        return d[j]->response();
      }
    return INFINITY;
  }

/* Return the octave of the i-th (or last) interest point that is described using 'type'.
   If there are no interest points that use this type, return infinity. */
signed int Signature::octave(unsigned int index, unsigned char type) const
  {
    unsigned int i, j;
    if(descriptorCtr[type] > 0)
      {
        i = 0;
        j = 0;
        while(i < _numPoints && !(j == index && d[j]->type() == type))
          {
            if(d[j]->type() == type)
              j++;
            i++;
          }
        return d[j]->octave();
      }
    return INT_MAX;
  }

#endif