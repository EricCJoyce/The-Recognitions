#include <Python.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#if PY_MAJOR_VERSION < 3
#error "Requires Python 3"
#include "stopcompilation"
#endif

#define ROW(i, w)  ((i - (i % w)) / w)                              /* On which row is index 'i'? */
#define COL(i, w)  (i % w)                                          /* In which column is index 'i'? */

/*
#define __DTW_DEBUG 1
*/

unsigned int build_L2_matrix(double*, unsigned int, double*, unsigned int, unsigned int, double**);
unsigned int viterbi(unsigned int, unsigned int, double*, double*, unsigned int**, unsigned int**, unsigned int**);

/* Given two matrices, Q in Real^q_len-by-d, T in Real^t_len-by-d, compute the L2 distance between
   all pairs of elemtns. Both Q (for query) and T (for template) are sequences of encoded frames.
   Each row in Q and in T is a vector derived from an enactment, and d, the number of columns, is the
   dimensionality of the encoding vector.
   This function writes the new values into the given pointer 'C' (for cost) and returns the length
   of that array--which we already know will be equal to q_len * t_len.
   Matrix C is written ROW-MAJOR with q_len rows and t_len columns. */
unsigned int build_L2_matrix(double* Q, unsigned int q_len,
                             double* T, unsigned int t_len,
                             unsigned int d, double** C)
  {
    unsigned int len = q_len * t_len;
    unsigned int x, y, i;
    double diff, accum;

    if(((*C) = (double*)malloc(len * sizeof(double))) == NULL)      //  Allocate the cost matrix to which we will write.
      return 0;

    for(y = 0; y < q_len; y++)                                      //  For every frame of the query...
      {
        for(x = 0; x < t_len; x++)                                  //  For every frame of the template...
          {
            accum = 0.0;                                            //  (Re)set an accumulator.
            for(i = 0; i < d; i++)                                  //  For every element in the two vectors...
              {
                diff = Q[y * d + i] - T[x * d + i];                 //  Store the difference between elements
                accum += diff * diff;                               //  and add the square of differences to the accumulator.
              }
            (*C)[y * t_len + x] = sqrt(accum);                      //  C stores square roots, row-major.
          }
      }

    return len;                                                     //  Return the length of the array.
  }

/*  Wrapper for the above.
    Receives a list of tuples of floats: the query.
    Receives a list of tuples of floats: the template.
    For both query and template, the inner lists (of floats) must have the same dimension, call it d.
    For our purposes, the length of the outer lists (of lists) will have the same length, too--though that is not enforced.
    Returns a list of lists of Python floats, which are C doubles. */
static PyObject* L2(PyObject* Py_UNUSED(self), PyObject* args)
  {
    PyObject* Q;                                                    //  As recevied from Python, a list of tuples of floats.
    Py_ssize_t q_len;                                               //  Length of outer list = number of frames in query snippet.
    PyObject* T;                                                    //  As recevied from Python, a list of tuples of floats.
    Py_ssize_t t_len;                                               //  Length of outer list = number of frames in template snippet.

    PyObject* sublist;                                              //  Used to iterate over the lists of tuples of floats.
    Py_ssize_t sublist_len;
    Py_ssize_t i, j;

    double* query;                                                  //  The query as a row-major array, frames-by-dimensionality.
    unsigned int query_len;                                         //  Length of the outer list of Q.
    double* template;                                               //  The template as a row-major array, frames-by-dimensionality.
    unsigned int template_len;                                      //  Length of the outer list of T.
    unsigned int d = 0;                                             //  Dimensionality of each frame's vector.
    unsigned int ctr;
    double* C;                                                      //  To become the cost matrix.
    bool first = true;                                              //  The first pass over Q tells us what the dimensionality should be.

    PyObject* ret;                                                  //  The PyObject to be returned.

    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &Q, &PyList_Type, &T))
      return NULL;

    q_len = PyList_Size(Q);                                         //  Save list size: number of frames in Q
    t_len = PyList_Size(T);                                         //  Save list size: number of frames in T
    query_len = (unsigned int)q_len;                                //  Convert to unsigned ints for use in the C-side function.
    template_len = (unsigned int)t_len;

    for(i = 0; i < q_len; i++)                                      //  Iterate over Q; make sure it is a list of lists of floats.
      {
        sublist = PyList_GetItem(Q, i);

        if(!PyTuple_Check(sublist))
          {
            PyErr_SetString(PyExc_TypeError, "List must contain tuples of floats");
            return NULL;
          }

        sublist_len = PyTuple_Size(sublist);
        if(first)
          {
            first = false;
            d = (unsigned int)sublist_len;
          }
        else if((unsigned int)sublist_len != d)
          {
            PyErr_SetString(PyExc_TypeError, "All inner tuples of floats must have the same length");
            return NULL;
          }

        if(PyErr_Occurred())
          return NULL;
      }

    for(i = 0; i < t_len; i++)                                      //  Iterate over T; make sure it is a list of lists of floats.
      {
        sublist = PyList_GetItem(T, i);

        if(!PyTuple_Check(sublist))
          {
            PyErr_SetString(PyExc_TypeError, "List must contain tuples of floats");
            return NULL;
          }

        sublist_len = PyTuple_Size(sublist);
        if((unsigned int)sublist_len != d)
          {
            PyErr_SetString(PyExc_TypeError, "All inner tuples of floats must have the same length");
            return NULL;
          }

        if(PyErr_Occurred())
          return NULL;
      }

    if((query = (double*)malloc(query_len * d * sizeof(double))) == NULL)
      {
        PyErr_SetString(PyExc_ValueError, "Unable to allocate memory for query");
        return NULL;
      }

    if((template = (double*)malloc(template_len * d * sizeof(double))) == NULL)
      {
        PyErr_SetString(PyExc_ValueError, "Unable to allocate memory for template");
        return NULL;
      }

    ctr = 0;
    for(i = 0; i < q_len; i++)                                      //  Iterate over Q; fill values into the C-side matrix.
      {
        sublist = PyList_GetItem(Q, i);
        sublist_len = PyTuple_Size(sublist);

        for(j = 0; j < sublist_len; j++)
          {
                                                                    //  Fill in the query, row-major.
            query[ctr] = PyFloat_AsDouble(PyTuple_GetItem(sublist, j));
            ctr++;

            if(PyErr_Occurred())
              return NULL;
          }
      }

    ctr = 0;
    for(i = 0; i < t_len; i++)                                      //  Iterate over T; fill values into the C-side matrix.
      {
        sublist = PyList_GetItem(T, i);
        sublist_len = PyTuple_Size(sublist);

        for(j = 0; j < sublist_len; j++)
          {
                                                                    //  Fill in the template, row-major.
            template[ctr] = PyFloat_AsDouble(PyTuple_GetItem(sublist, j));
            ctr++;

            if(PyErr_Occurred())
              return NULL;
          }
      }
                                                                    //  Build the cost matrix.
    build_L2_matrix(query, query_len, template, template_len, d, &C);
                                                                    //  Pack contents of C into PyObject.
    ret = PyList_New(q_len);                                        //  Create a return object: as many rows (outer lists) as there are in Q.
    if(!ret)                                                        //  If it failed, clean up before we die.
      {
        PyErr_NoMemory();
        free(query);
        free(template);
        free(C);
        return NULL;
      }

    for(i = 0; i < q_len; i++)                                      //  For each row in Q, fill in as many values as there are rows in T.
      {
        PyList_SetItem(ret, i, PyList_New(t_len));                  //  Add a list at position i.
        for(j = 0; j < t_len; j++)                                  //  Add a float at position[i][j].
          PyList_SetItem(PyList_GetItem(ret, i), j, PyFloat_FromDouble(C[(unsigned int)i * template_len + (unsigned int)j]));
      }

    free(query);                                                    //  Clean up C-side memory allocations.
    free(template);
    free(C);

    if(PyErr_Occurred())                                            //  If something still went wrong, flag the return object
      {                                                             //  for garbage collection.
        Py_XDECREF(ret);
        return NULL;
      }

    return ret;
  }

/* Given a cost matrix, C in Real^rows-by-cols, compute the cheapest path
   from the last frames of both query and template to the first frames of both query and template.
   Store the cost of this path in (*cost).
   Stroe the path's indices (row-major) in (*path).
   Store the time-warped indices of the query in (*q).
   Store the time-warped indices of the template in (*t).
   Return the length of the path. */
unsigned int viterbi(unsigned int rows, unsigned int cols, double* C,
                     double* cost, unsigned int** path, unsigned int** q, unsigned int** t)
  {
    double* T_1;                                                    //  Hold accumulated costs so far.
    unsigned int* T_2;                                              //  Hold indices preferred so far.

    unsigned int len = 0;                                           //  Length of the cheapest path.
    unsigned int index, neighbor;
    unsigned int i, j;
    bool left_exists, up_exists;

    if((T_1 = (double*)malloc(rows * cols * sizeof(double))) == NULL)
      return 0;
    if((T_2 = (unsigned int*)malloc(rows * cols * sizeof(int))) == NULL)
      {
        free(T_1);
        return 0;
      }
    for(i = 0; i < rows * cols; i++)                                //  "Blank out" matrices.
      {
        T_1[i] = INFINITY;
        T_2[i] = UINT_MAX;
      }

    for(i = 0; i < rows; i++)
      {
        for(j = 0; j < cols; j++)
          {
            index = i * cols + j;
            if(index == 0)
              T_1[index] = C[index];
            else
              {
                left_exists = (COL(index, cols) > 0);               //  Does another cell exist to the left of this one?
                up_exists = (ROW(index, cols) > 0);                 //  Does another cell exist above this one?

                if(left_exists && up_exists)                        //  Compare cost of arriving at 'index' from 'neighbor' = above-left.
                  {
                    neighbor = index - cols - 1;
                    if(2.0 * C[index] + T_1[neighbor] < T_1[index])
                      {
                        T_1[index] = 2.0 * C[index] + T_1[neighbor];
                        T_2[index] = neighbor;
                      }
                  }

                if(left_exists)                                     //  Compare cost of arriving at 'index' from 'neighbor' = left.
                  {
                    neighbor = index - 1;
                    if(C[index] + T_1[neighbor] < T_1[index])
                      {
                        T_1[index] = C[index] + T_1[neighbor];
                        T_2[index] = neighbor;
                      }
                  }

                if(up_exists)                                       //  Compare cost of arriving at 'index' from 'neighbor' = above.
                  {
                    neighbor = index - cols;
                    if(C[index] + T_1[neighbor] < T_1[index])
                      {
                        T_1[index] = C[index] + T_1[neighbor];
                        T_2[index] = neighbor;
                      }
                  }
              }
          }
      }

    index = rows * cols - 1;                                        //  Count up.
    i = 0;
    (*cost) = 0.0;
    while(index != UINT_MAX)
      {
        i++;
        (*cost) += T_1[index];
        index = T_2[index];
      }
    (*cost) /= (double)(rows + cols);                               //  Normalize by (M + N).
    len = i;                                                        //  Save path length.
                                                                    //  Allocate.
    if(((*path) = (unsigned int*)malloc(len * sizeof(int))) == NULL)
      {
        free(T_1);
        free(T_2);
        return 0;
      }
    if(((*q) = (unsigned int*)malloc(len * sizeof(int))) == NULL)
      {
        free(T_1);
        free(T_2);
        free(path);
        return 0;
      }
    if(((*t) = (unsigned int*)malloc(len * sizeof(int))) == NULL)
      {
        free(T_1);
        free(T_2);
        free(path);
        free(q);
        return 0;
      }

    index = rows * cols - 1;                                        //  Reset.
    i = 0;
    while(index != UINT_MAX)
      {
        (*path)[len - i - 1] = index;
        (*q)[len - i - 1] = COL(index, cols);
        (*t)[len - i - 1] = ROW(index, cols);

        i++;
        index = T_2[index];
      }

    free(T_1);                                                      //  Clean up. Go home.
    free(T_2);

    return len;
  }

/* Receives a list of lists of floats: the cost matrix.
   Returns four objects (in a 4-tuple):
   1.) The cost of the cheapest path   (float)
   2.) The path itself                 (tuple of ints)
   3.) The sequence of Query frames    (tuple of ints)
   4.) The sequence of Template frames (tuple of ints)  */
static PyObject* path(PyObject* Py_UNUSED(self), PyObject* args)
  {
    PyObject* C;                                                    //  As recevied from Python, a list of lists of floats.
    Py_ssize_t c_rows;                                              //  Length of outer list = number of frames in query snippet.

    PyObject* sublist;                                              //  Used to iterate over the lists of lists of floats.
    Py_ssize_t sublist_len;
    Py_ssize_t i, j;

    double* cost;                                                   //  The cost matrix as a row-major array, length of Q by length of T.
    unsigned int cost_rows, cost_cols = 0;                          //  Same dimensions as above, but as uints rather than Py_ssize_ts.
    double total_cost;                                              //  Total cost of the cheapest path.
    unsigned int* cost_path;                                        //  Array of row-major indices into the cost matrix.
    unsigned int pathLen = 0;
    unsigned int* alignment_a;                                      //  Array of frame indices into the QUERY.
    unsigned int* alignment_b;                                      //  Array of frame indices into the TEMPLATE.
    unsigned int ctr;
    bool first = true;

    PyObject* ret;                                                  //  The PyObject to be returned.

    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &C))
      return NULL;

    c_rows = PyList_Size(C);                                        //  Save list size: number of frames in Q
    cost_rows = (unsigned int)c_rows;                               //  Convert to unsigned ints for use in the C-side function.

    for(i = 0; i < c_rows; i++)                                     //  Iterate over C; make sure it is a list of lists of floats.
      {
        sublist = PyList_GetItem(C, i);

        if(!PyList_Check(sublist))
          {
            PyErr_SetString(PyExc_TypeError, "List must contain lists of floats");
            return NULL;
          }

        sublist_len = PyList_Size(sublist);
        if(first)
          {
            first = false;
            cost_cols = (unsigned int)sublist_len;
          }
        else if((unsigned int)sublist_len != cost_cols)
          {
            PyErr_SetString(PyExc_TypeError, "All inner lists of floats must have the same length");
            return NULL;
          }

        if(PyErr_Occurred())
          return NULL;
      }

    if((cost = (double*)malloc(cost_rows * cost_cols * sizeof(double))) == NULL)
      {
        PyErr_SetString(PyExc_ValueError, "Unable to allocate memory for cost matrix");
        return NULL;
      }

    ctr = 0;
    for(i = 0; i < c_rows; i++)                                     //  Iterate over C; now filling in its values.
      {
        sublist = PyList_GetItem(C, i);
        sublist_len = PyList_Size(sublist);

        for(j = 0; j < sublist_len; j++)
          {
                                                                    //  Fill in the costs, row-major.
            cost[ctr] = PyFloat_AsDouble(PyList_GetItem(sublist, j));
            ctr++;

            if(PyErr_Occurred())
              return NULL;
          }
      }

    pathLen = viterbi(cost_rows, cost_cols, cost, &total_cost, &cost_path, &alignment_a, &alignment_b);

    ret = PyTuple_New(4);                                           //  Create a return object: a 4-tuple.
    if(!ret)                                                        //  If it failed, clean up before we die.
      {
        PyErr_NoMemory();
        free(cost);
        if(pathLen > 0)
          {
            free(cost_path);
            free(alignment_a);
            free(alignment_b);
          }
        return NULL;
      }

    PyTuple_SetItem(ret, 0, PyFloat_FromDouble(total_cost));        //  Set element zero: the total cost of the cheapest path.

    PyTuple_SetItem(ret, 1, PyTuple_New(pathLen));                  //  Set element one: a tuple to contain the cost-matrix elements.
    for(i = 0; i < pathLen; i++)
      PyTuple_SetItem(PyTuple_GetItem(ret, 1), i, PyLong_FromLong(cost_path[i]));

    PyTuple_SetItem(ret, 2, PyTuple_New(pathLen));                  //  Set element two: a tuple to contain Query frame indices.
    for(i = 0; i < pathLen; i++)
      PyTuple_SetItem(PyTuple_GetItem(ret, 2), i, PyLong_FromLong(alignment_a[i]));

    PyTuple_SetItem(ret, 3, PyTuple_New(pathLen));                  //  Set element three: a tuple to contain Template frame indices.
    for(i = 0; i < pathLen; i++)
      PyTuple_SetItem(PyTuple_GetItem(ret, 3), i, PyLong_FromLong(alignment_b[i]));

    free(cost);                                                     //  Clean up C-side memory allocations.
    if(pathLen > 0)
      {
        free(cost_path);
        free(alignment_a);
        free(alignment_b);
      }

    if(PyErr_Occurred())                                            //  If something still went wrong, flag the return object
      {                                                             //  for garbage collection.
        Py_XDECREF(ret);
        return NULL;
      }

    return ret;
  }

/* This function handles the tasks of both L2() and path() in one call. */
static PyObject* DTW(PyObject* Py_UNUSED(self), PyObject* args)
  {
    PyObject* Q;                                                    //  As recevied from Python, a list of tuples of floats.
    Py_ssize_t q_len;                                               //  Length of outer list = number of frames in query snippet.
    PyObject* T;                                                    //  As recevied from Python, a list of tuples of floats.
    Py_ssize_t t_len;                                               //  Length of outer list = number of frames in template snippet.

    PyObject* sublist;                                              //  Used to iterate over the lists of tuples of floats.
    Py_ssize_t sublist_len;
    Py_ssize_t i, j;

    double* query;                                                  //  The query as a row-major array, frames-by-dimensionality.
    unsigned int query_len;                                         //  Length of the outer list of Q.
    double* template;                                               //  The template as a row-major array, frames-by-dimensionality.
    unsigned int template_len;                                      //  Length of the outer list of T.
    unsigned int d = 0;                                             //  Dimensionality of each frame's vector.
    unsigned int ctr;
    double* C;                                                      //  To become the cost matrix.
    bool first = true;                                              //  The first pass over Q tells us what the dimensionality should be.

    double total_cost;                                              //  Total cost of the cheapest path.
    unsigned int* cost_path;                                        //  Array of row-major indices into the cost matrix.
    unsigned int pathLen = 0;
    unsigned int* alignment_a;                                      //  Array of frame indices into the QUERY.
    unsigned int* alignment_b;                                      //  Array of frame indices into the TEMPLATE.

    PyObject* ret;                                                  //  The PyObject to be returned.

    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &Q, &PyList_Type, &T))
      return NULL;

    q_len = PyList_Size(Q);                                         //  Save list size: number of frames in Q
    t_len = PyList_Size(T);                                         //  Save list size: number of frames in T
    query_len = (unsigned int)q_len;                                //  Convert to unsigned ints for use in the C-side function.
    template_len = (unsigned int)t_len;

    for(i = 0; i < q_len; i++)                                      //  Iterate over Q; make sure it is a list of tuples of floats.
      {
        sublist = PyList_GetItem(Q, i);

        if(!PyTuple_Check(sublist))
          {
            PyErr_SetString(PyExc_TypeError, "List must contain tuples of floats");
            return NULL;
          }

        sublist_len = PyTuple_Size(sublist);
        if(first)
          {
            first = false;
            d = (unsigned int)sublist_len;
          }
        else if((unsigned int)sublist_len != d)
          {
            PyErr_SetString(PyExc_TypeError, "All inner tuples of floats must have the same length");
            return NULL;
          }

        if(PyErr_Occurred())
          return NULL;
      }

    for(i = 0; i < t_len; i++)                                      //  Iterate over T; make sure it is a list of tuples of floats.
      {
        sublist = PyList_GetItem(T, i);

        if(!PyTuple_Check(sublist))
          {
            PyErr_SetString(PyExc_TypeError, "List must contain tuples of floats");
            return NULL;
          }

        sublist_len = PyTuple_Size(sublist);
        if((unsigned int)sublist_len != d)
          {
            PyErr_SetString(PyExc_TypeError, "All inner tuples of floats must have the same length");
            return NULL;
          }

        if(PyErr_Occurred())
          return NULL;
      }

    if((query = (double*)malloc(query_len * d * sizeof(double))) == NULL)
      {
        PyErr_SetString(PyExc_ValueError, "Unable to allocate memory for query");
        return NULL;
      }

    if((template = (double*)malloc(template_len * d * sizeof(double))) == NULL)
      {
        PyErr_SetString(PyExc_ValueError, "Unable to allocate memory for template");
        return NULL;
      }

    ctr = 0;
    for(i = 0; i < q_len; i++)                                      //  Iterate over Q; fill values into the C-side matrix.
      {
        sublist = PyList_GetItem(Q, i);
        sublist_len = PyTuple_Size(sublist);

        for(j = 0; j < sublist_len; j++)
          {
                                                                    //  Fill in the query, row-major.
            query[ctr] = PyFloat_AsDouble(PyTuple_GetItem(sublist, j));
            ctr++;

            if(PyErr_Occurred())
              return NULL;
          }
      }

    ctr = 0;
    for(i = 0; i < t_len; i++)                                      //  Iterate over T; fill values into the C-side matrix.
      {
        sublist = PyList_GetItem(T, i);
        sublist_len = PyTuple_Size(sublist);

        for(j = 0; j < sublist_len; j++)
          {
                                                                    //  Fill the template, row-major.
            template[ctr] = PyFloat_AsDouble(PyTuple_GetItem(sublist, j));
            ctr++;

            if(PyErr_Occurred())
              return NULL;
          }
      }
                                                                    //  Build the cost matrix.
    build_L2_matrix(query, query_len, template, template_len, d, &C);

    pathLen = viterbi(query_len, template_len, C, &total_cost, &cost_path, &alignment_a, &alignment_b);

    ret = PyTuple_New(4);                                           //  Create a return object: a 4-tuple.
    if(!ret)                                                        //  If it failed, clean up before we die.
      {
        PyErr_NoMemory();
        free(C);
        if(pathLen > 0)
          {
            free(cost_path);
            free(alignment_a);
            free(alignment_b);
          }
        return NULL;
      }

    PyTuple_SetItem(ret, 0, PyFloat_FromDouble(total_cost));        //  Set element zero: the total cost of the cheapest path.

    PyTuple_SetItem(ret, 1, PyTuple_New(pathLen));                  //  Set element one: a tuple to contain the cost-matrix elements.
    for(i = 0; i < pathLen; i++)
      PyTuple_SetItem(PyTuple_GetItem(ret, 1), i, PyLong_FromLong(cost_path[i]));

    PyTuple_SetItem(ret, 2, PyTuple_New(pathLen));                  //  Set element two: a tuple to contain Query frame indices.
    for(i = 0; i < pathLen; i++)
      PyTuple_SetItem(PyTuple_GetItem(ret, 2), i, PyLong_FromLong(alignment_a[i]));

    PyTuple_SetItem(ret, 3, PyTuple_New(pathLen));                  //  Set element three: a tuple to contain Template frame indices.
    for(i = 0; i < pathLen; i++)
      PyTuple_SetItem(PyTuple_GetItem(ret, 3), i, PyLong_FromLong(alignment_b[i]));

    free(C);                                                        //  Clean up C-side memory allocations.
    if(pathLen > 0)
      {
        free(cost_path);
        free(alignment_a);
        free(alignment_b);
      }

    if(PyErr_Occurred())                                            //  If something still went wrong, flag the return object
      {                                                             //  for garbage collection.
        Py_XDECREF(ret);
        return NULL;
      }

    return ret;
  }

static PyMethodDef methods[] =
  {
    {"L2", &L2, METH_VARARGS, "Compute the cost matrix between all frames using L2 distance.\nInput: two lists of tuples of floats, Q and T. The number of columns must be the same.\nOutput: a matrix with as many rows as Q has frames and as many columns as T has frames."},
    {"path", &path, METH_VARARGS, "Compute the alignment path through a given cost matrix.\nInput: cost matrix as a list of lists of floats (number of Q frames by number of T frames).\nOutput: the cost of the path; the path itself indexing into the cost matrix; alignment of Query frames; alignment of Template frames."},
    {"DTW", &DTW, METH_VARARGS, "Compute the alignment for a given Query and a given Template.\nInput: query as a list of tuples of floats (number of Q frames by vector length); template as a list of tuples of floats (number of T frames by vector length).\nOutput: the cost of the path; the path itself indexing into the cost matrix; alignment of Query frames; alignment of Template frames."},
    {NULL, NULL, 0, NULL}
  };

static struct PyModuleDef module_def =
  {
    PyModuleDef_HEAD_INIT,                                          //  Always required
    "DTW",                                                          //  Module name
    "Perform dynamic time warping to align query and template sequences",
    -1,                                                             //  Module size (-1 indicates we don't use this feature)
    methods,                                                        //  Method table
  };

PyMODINIT_FUNC PyInit_DTW(void)
  {
    return PyModule_Create(&module_def);
  }
