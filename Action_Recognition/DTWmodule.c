#include <Python.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#if PY_MAJOR_VERSION < 3
#error "Requires Python 3"
#include "stopcompilation"
#endif

#define PARENT(i)  ((i - 1) / 2)                                    /* Return index of parent in heap. */
#define LEFT(i)    (2 * i) + 1                                      /* Return index of left child in heap. */
#define RIGHT(i)   (2 * i) + 2                                      /* Return index of right child in heap. */
#define SWAP(a, b) ({a ^= b; b ^= a; a ^= b;})                      /* Swap heap indices. */
#define MIN(a, b)  (a < b ? a : b)                                  /* Compare rise to run (unsigned ints). */

unsigned int build_L2_matrix(double*, unsigned int, double*, unsigned int, unsigned int, double**);
unsigned int a_star(unsigned int, unsigned int, double*, double*, unsigned int**, unsigned int**, unsigned int**);
double heuristic(unsigned int, unsigned int);
void reconstruct_path(unsigned int**, unsigned int*, unsigned int**, unsigned int);
bool found(unsigned int**, unsigned int*, unsigned int);
void heapify(unsigned int**, unsigned int*, unsigned int, double**);
void insert_heap(unsigned int**, unsigned int*, unsigned int, double**);
unsigned int extract_min(unsigned int**, unsigned int*, double**);
void decrease_key(unsigned int**, unsigned int*, unsigned int);
void delete_key(unsigned int**, unsigned int*, unsigned int, double**);

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
/*
unsigned int viterbi(double* C, unsigned int rows, unsigned int cols,
                     double* cost, double** path, double** q, double** t)
  {
    double* T_1;                                                    //  Hold accumulated costs so far.
    unsigned int* T_2;                                              //  Hold indices preferred so far.
    unsigned int T = rows + cols;

    unsigned int index, neighbor;
    double val, min_val;
    unsigned int min_index;
    unsigned int len = 0;                                           //  Length of the path.
    unsigned int i, j;
    bool up_exists, left_exists;
                                                                    //  Longest possible paths would be along the edges.
    if((T_1 = (double*)malloc(3 * T * sizeof(double))) == NULL)     //  At most three possible transitions: up, left, up-left.
      return 0;
    if((T_2 = (unsigned int*)malloc(3 * T * sizeof(int))) == NULL)
      {
        free(T_1);
        return 0;
      }
    for(i = 0; i < 3; i++)                                          //  Initialize T_1 and T_2.
      {
        T_1[i * (rows + cols)] = 0.0;                               //  Initialize row-major, first column costs.
        T_2[i * (rows + cols)] = rows * cols - 1;                   //  Initialize row-major, first column sources.
      }
    for(j = 1; j < T; j++)                                          //  Longest possible path = T = half cost perimeter.
      {
        min_val = INFINITY;
        min_index = UINT_MAX;

        for(i = 0; i < 3; i++)                                      //  At most three possible "states": left, up, up-left.
          {
            index = T_2[i * T + j - 1];

            up_exists = ((index - (index % rows)) / rows > 0);
            left_exists = (index % rows > 0);

            switch(i)
              {
                case 0: if(up_exists && left_exists)                //  Up-left
                          {
                            neighbor = index - cols - 1;
                            val = 2.0 * (C[neighbor] + C[index]);
                          }
                        else
                          {
                            neighbor = UINT_MAX;
                            val = INFINITY;
                          }
                        break;
                case 1: if(left_exists)                             //  Left
                          {
                            neighbor = index - 1;
                            val = C[neighbor] + C[index];
                          }
                        else
                          {
                            neighbor = UINT_MAX;
                            val = INFINITY;
                          }
                        break;
                case 2: if(up_exists)                               //  Up
                          {
                            neighbor = index - cols;
                            val = C[neighbor] + C[index];
                          }
                        else
                          {
                            neighbor = UINT_MAX;
                            val = INFINITY;
                          }
                        break;
              }

            T_1[i * T + j] = min_val;
            T_2[i * T + j] = min_index;
          }
      }

    return len;
  }
*/
/* Screw it. Use A*. */
unsigned int a_star(unsigned int rows, unsigned int cols, double* C, double* cost,
                    unsigned int** path, unsigned int** alignment_a, unsigned int** alignment_b)
  {
    unsigned int i;
    unsigned int start;                                             //  'goal' is always zero (element [0, 0]).
    unsigned int current_index, neighbor;
    bool left_exists, above_exists;                                 //  Test whether cells exist to the left or above the current matrix cell.

    unsigned int* came_from;                                        //  Matrix of source indices: came_from[n] is the index of the node
                                                                    //  immediately preceding n on the current cheapest path from start to n.
    double* G;                                                      //  For node n, G[n] is the cost of the current cheapest path from start to n.
    double* F;                                                      //  For node n, F[n] = G[n] + heuristic(n).
    double g;                                                       //  F[n] represents our current best guess as to how short a path from start
                                                                    //  to finish can be if it goes through n.
    unsigned int pathLen = 0;                                       //  Value to be returned.

    unsigned int* heap;                                             //  This will be "over"-allocated.
    unsigned int heapLen = 0;                                       //  The effective size of the heap, <= total capacity.
                                                                    //  Allocate source-cell lookup.
    if((came_from = (unsigned int*)malloc(rows * cols * sizeof(int))) == NULL)
      exit(1);

    if((F = (double*)malloc(rows * cols * sizeof(double))) == NULL) //  Allocate estimated cost matrix.
      exit(1);

    if((G = (double*)malloc(rows * cols * sizeof(double))) == NULL) //  Allocate known cost matrix.
      exit(1);

    for(i = 0; i < rows * cols; i++)                                //  Initialize.
      {
        came_from[i] = UINT_MAX;
        G[i] = INFINITY;                                            //  Fill in costs.
        F[i] = INFINITY;                                            //  Fill in estimated costs.
      }
                                                                    //  (Over)allocate the heap (and avoid realloc()).
    if((heap = (unsigned int*)malloc(rows * cols * sizeof(int))) == NULL)
      exit(1);

    start = rows * cols - 1;                                        //  Algorithm starts in the lower-right.
                                                                    //  Algorithm ends in the upper-left (at zero).
    heap[0] = start;
    heapLen++;
    G[start] = 0.0;
    F[start] = heuristic(start, cols);

    while(heapLen > 0)                                              //  Begin A*
      {
        current_index = extract_min(&heap, &heapLen, &F);
        if(current_index == 0)                                      //  Goal reached!
          {
            reconstruct_path(path, &pathLen, &came_from, start);    //  Build path.

            if(((*alignment_a) = (unsigned int*)malloc(pathLen * sizeof(int))) == NULL)
              exit(1);
            if(((*alignment_b) = (unsigned int*)malloc(pathLen * sizeof(int))) == NULL)
              exit(1);

            (*cost) = 0.0;
            for(i = 0; i < pathLen; i++)                            //  Build alignment sequences.
              {
                                                                    //  'a' receives the ROWS.
                (*alignment_a)[i] = ((*path)[i] - ((*path)[i] % cols)) / cols;
                (*alignment_b)[i] = (*path)[i] % cols;              //  'b' receives the COLUMNS.
                (*cost) += G[ (*path)[i] ];                         //  Add up cost.
              }
            (*cost) /= (double)(rows + cols);                       //  Normalize cost by N + M.

            break;
          }

        left_exists = (current_index % cols > 0);                   //  A cell exists to the left.
                                                                    //  A cell exists above.
        above_exists = ((current_index - (current_index % cols)) / rows > 0);

        if(left_exists && above_exists)                             //  Diagonal to the upper-left exists.
          {
            neighbor = current_index - cols - 1;
            g = 2.0 * (G[current_index] + C[neighbor]);

            if(g < G[neighbor])
              {
                came_from[neighbor] = current_index;
                G[neighbor] = g;
                F[neighbor] = g + heuristic(neighbor, cols);
                if(!found(&heap, &heapLen, neighbor))
                  insert_heap(&heap, &heapLen, neighbor, &F);
              }
          }

        if(left_exists)                                             //  Left exists.
          {
            neighbor = current_index - 1;
            g = (G[current_index] + C[neighbor]);

            if(g < G[neighbor])
              {
                came_from[neighbor] = current_index;
                G[neighbor] = g;
                F[neighbor] = g + heuristic(neighbor, cols);
                if(!found(&heap, &heapLen, neighbor))
                  insert_heap(&heap, &heapLen, neighbor, &F);
              }
          }

        if(above_exists)                                            //  Above exists.
          {
            neighbor = current_index - cols;
            g = (G[current_index] + C[neighbor]);

            if(g < G[neighbor])
              {
                came_from[neighbor] = current_index;
                G[neighbor] = g;
                F[neighbor] = g + heuristic(neighbor, cols);
                if(!found(&heap, &heapLen, neighbor))
                  insert_heap(&heap, &heapLen, neighbor, &F);
              }
          }
      }

    free(came_from);                                                //  Clean up. Go home.
    free(G);
    free(F);
    free(heap);

    return pathLen;
  }

/* Estimated cost is the Manhattan distance to goal. */
double heuristic(unsigned int index, unsigned int cols)
  {
    return (double)(MIN((index - (index % cols)) / cols), (index % cols));
  }

void reconstruct_path(unsigned int** path, unsigned int* pathLen, unsigned int** came_from, unsigned int start)
  {
    unsigned int p = 0;                                             //  Goal is always 0.
    unsigned int ctr = 1;

    while(p != start)                                               //  Pass 1: the count-up.
      {
        p = (*came_from)[p];
        ctr++;
      }

    (*pathLen) = ctr;                                               //  Now allocate.
    if(((*path) = (unsigned int*)malloc(ctr * sizeof(int))) == NULL)
      exit(1);

    p = 0;                                                          //  Reset.
    ctr = 0;
    while(p != start)                                               //  Pass 2: storage.
      {
        (*path)[ctr] = p;                                           //  Fill in path.
        ctr++;
        p = (*came_from)[p];
      }
    (*path)[ctr] = start;

    return;
  }

/* Is 'index' in the heap? */
bool found(unsigned int** heap, unsigned int* heapLen, unsigned int index)
  {
    unsigned int i = 0;

    while(i < (*heapLen) && (*heap)[i] != index)
      i++;
    return i < (*heapLen);
  }

/* Preserve heap properties, according to values in the cost matrix 'Mat'. */
void heapify(unsigned int** heap, unsigned int* heapLen, unsigned int index, double** Mat)
  {
    unsigned int left, right;
    unsigned int smallest;

    left = LEFT(index);
    right = RIGHT(index);
    smallest = index;

    if(left < (*heapLen) && (*Mat)[ (*heap)[left] ] < (*Mat)[ (*heap)[index] ])
      smallest = left;

    if(right < (*heapLen) && (*Mat)[ (*heap)[right] ] < (*Mat)[ (*heap)[smallest] ])
      smallest = right;

    if(smallest != index)
      {
        SWAP((*heap)[index], (*heap)[smallest]);
        heapify(heap, heapLen, smallest, Mat);
      }

    return;
  }

/* Add value 'new_index' to the given 'heap' with current effective capacity 'heapLen'.
   This function increases 'heapLen' and preserves the min-heap property. */
void insert_heap(unsigned int** heap, unsigned int* heapLen, unsigned int new_index, double** Mat)
  {
    unsigned int i = (*heapLen);

    (*heap)[(*heapLen)] = new_index;
    (*heapLen)++;

    while(i != 0 && (*Mat)[ (*heap)[PARENT(i)] ] > (*Mat)[ (*heap)[i] ])
      {
        SWAP((*heap)[i], (*heap)[PARENT(i)]);
        i = PARENT(i);
      }

    return;
  }

/* Pop the root value and repair the heap. */
unsigned int extract_min(unsigned int** heap, unsigned int* heapLen, double** Mat)
  {
    unsigned int root;

    if((*heapLen) == 0)
      return UINT_MAX;

    if((*heapLen) == 1)
      {
        (*heapLen)--;
        return (*heap)[0];
      }

    root = (*heap)[0];
    (*heap)[0] = (*heap)[(*heapLen) - 1];
    (*heapLen)--;
    heapify(heap, heapLen, 0, Mat);

    return root;
  }

/* Remove the node at index and repair the heap. */
void decrease_key(unsigned int** heap, unsigned int* heapLen, unsigned int index)
  {
    unsigned int i = index;

    (*heap)[i] = UINT_MAX;                                          //  I want to take advantage of unsigned int's range,
                                                                    //  so we will treat UINT_MAX like negative infinity.
    while(i != 0)
      {
        SWAP((*heap)[i], (*heap)[PARENT(i)]);
        i = PARENT(i);
      }

    return;
  }

/* Delete the given 'index' and repair the heap. */
void delete_key(unsigned int** heap, unsigned int* heapLen, unsigned int index, double** Mat)
  {
    decrease_key(heap, heapLen, index);
    extract_min(heap, heapLen, Mat);
    return;
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

    pathLen = a_star(cost_rows, cost_cols, cost, &total_cost, &cost_path, &alignment_a, &alignment_b);
    //pathLen = viterbi(cost_rows, cost_cols, cost, &total_cost, &cost_path, &alignment_a, &alignment_b);

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

    pathLen = a_star(query_len, template_len, C, &total_cost, &cost_path, &alignment_a, &alignment_b);
    //pathLen = viterbi(cost_rows, cost_cols, cost, &total_cost, &cost_path, &alignment_a, &alignment_b);

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
