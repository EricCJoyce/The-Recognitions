/**************************************************************************************************
 A*-Search for Dynamic Time-Warping, by Eric C. Joyce
 Program expects two matrix dimensions: number of rows, number of columns, and as many floats.
 Program computes the cheapest path from the lower-right corner to the upper-left corner and
 prints its solution to standard out.
***************************************************************************************************/

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define PARENT(i)  ((i - 1) / 2)                                    /* Return index of parent in heap. */
#define LEFT(i)    (2 * i) + 1                                      /* Return index of left child in heap. */
#define RIGHT(i)   (2 * i) + 2                                      /* Return index of right child in heap. */
#define SWAP(a, b) ({a ^= b; b ^= a; a ^= b;})                      /* Swap heap indices. */
/**/
#define __ALIGN_DEBUG 1
/**/

float heuristic(unsigned int);
void reconstruct_path(unsigned int**, unsigned int*, unsigned int**, unsigned int);

bool found(unsigned int**, unsigned int*, unsigned int);
void delete_key(unsigned int**, unsigned int*, unsigned int, float**);
unsigned int extract_min(unsigned int**, unsigned int*, float**);
void decrease_key(unsigned int**, unsigned int*, unsigned int);
void insert_heap(unsigned int**, unsigned int*, unsigned int, float**);
void heapify(unsigned int**, unsigned int*, unsigned int, float**);

void print_fmatrix(unsigned int, unsigned int, float**);
void print_umatrix(unsigned int, unsigned int, unsigned int**);
void print_variables(unsigned int, unsigned int, float**, float**, float**, unsigned int**);
void print_heap(unsigned int**, unsigned int*, float**);

int main(int argc, char* argv[])
  {
    unsigned int i = 1;
    float* C;                                                       //  Cost matrix
    unsigned int rows, cols;                                        //  Dimensions of cost matrix

    unsigned int start;                                             //  'goal' is always zero (element [0, 0]).
    unsigned int current_index, neighbor;
    bool left_exists, above_exists;                                 //  Test whether cells exist to the left or above the current matrix cell.

    unsigned int* came_from;                                        //  Matrix of source indices: came_from[n] is the index of the node
                                                                    //  immediately preceding n on the current cheapest path from start to n.
    float* G;                                                       //  For node n, G[n] is the cost of the current cheapest path from start to n.
    float* F;                                                       //  For node n, F[n] = G[n] + heuristic(n).
    float g;                                                        //  F[n] represents our current best guess as to how short a path from start
                                                                    //  to finish can be if it goes through n.
    unsigned int* path;
    unsigned int pathLen = 0;
    unsigned int* alignment_a;
    unsigned int* alignment_b;

    unsigned int* heap;                                             //  This will be "over"-allocated.
    unsigned int heapLen = 0;                                       //  The effective size of the heap, <= total capacity.

    while(i < (unsigned int)argc)                                   //  Read in matrix dimensions and contents.
      {
        if(i == 1)
          rows = (unsigned int)atoi(argv[i]);                       //  Save number of rows.
        else if(i == 2)
          {
           cols = (unsigned int)atoi(argv[i]);                      //  Save number of columns.
                                                                    //  Allocate the cost matrix.
           if((C = (float*)malloc(rows * cols * sizeof(float))) == NULL)
             exit(1);
                                                                    //  Allocate the came-from matrix.
           if((came_from = (unsigned int*)malloc(rows * cols * sizeof(int))) == NULL)
             exit(1);
                                                                    //  Allocate the (known) cost matrix.
           if((G = (float*)malloc(rows * cols * sizeof(float))) == NULL)
             exit(1);
                                                                    //  Allocate the (estimated) cost matrix.
           if((F = (float*)malloc(rows * cols * sizeof(float))) == NULL)
             exit(1);
                                                                    //  (Over)allocate the heap (and avoid realloc()).
           if((heap = (unsigned int*)malloc(rows * cols * sizeof(int))) == NULL)
             exit(1);
          }
        else
          {
            C[i - 3] = (float)atof(argv[i]);                        //  Fill in the cost matrix, row-major.
            came_from[i - 3] = UINT_MAX;
            G[i - 3] = INFINITY;                                    //  Fill in costs.
            F[i - 3] = INFINITY;                                    //  Fill in estimated costs.
          }
        i++;
      }

    start = rows * cols - 1;                                        //  Algorithm starts in the lower-right.
                                                                    //  Algorithm ends in the upper-left (at zero).
    heap[0] = start;
    heapLen++;
    G[start] = 0.0;
    F[start] = heuristic(start);

    while(heapLen > 0)                                              //  Begin A*
      {
        #ifdef __ALIGN_DEBUG
        printf("\n>>>\n");
        print_heap(&heap, &heapLen, &F);
        print_variables(rows, cols, &C, &F, &G, &came_from);
        #endif

        current_index = extract_min(&heap, &heapLen, &F);
        if(current_index == 0)                                      //  Goal reached!
          {
            reconstruct_path(&path, &pathLen, &came_from, start);   //  Build path.

            if((alignment_a = (unsigned int*)malloc(pathLen * sizeof(int))) == NULL)
              exit(1);
            if((alignment_b = (unsigned int*)malloc(pathLen * sizeof(int))) == NULL)
              exit(1);

            for(i = 0; i < pathLen; i++)                            //  Build alignment sequences.
              {
                                                                    //  'a' receives the ROWS.
                alignment_a[i] = (path[i] - (path[i] % cols)) / cols;
                alignment_b[i] = path[i] % cols;                    //  'b' receives the COLUMNS.
              }

            break;
          }

        left_exists = (current_index % cols > 0);                   //  A cell exists to the left.
                                                                    //  A cell exists above.
        above_exists = ((current_index - (current_index % cols)) / rows >= 0);

        if(left_exists && above_exists)                             //  Diagonal to the upper-left exists.
          {
            neighbor = current_index - cols - 1;
            g = G[current_index] + C[neighbor];

            if(g < G[neighbor])
              {
                came_from[neighbor] = current_index;
                G[neighbor] = g;
                F[neighbor] = g + heuristic(neighbor);
                if(!found(&heap, &heapLen, neighbor))
                  insert_heap(&heap, &heapLen, neighbor, &F);
              }
          }

        if(left_exists)                                             //  Left exists.
          {
            neighbor = current_index - 1;
            g = G[current_index] + C[neighbor];

            if(g < G[neighbor])
              {
                came_from[neighbor] = current_index;
                G[neighbor] = g;
                F[neighbor] = g + heuristic(neighbor);
                if(!found(&heap, &heapLen, neighbor))
                  insert_heap(&heap, &heapLen, neighbor, &F);
              }
          }

        if(above_exists)                                            //  Above exists.
          {
            neighbor = current_index - cols;
            g = G[current_index] + C[neighbor];

            if(g < G[neighbor])
              {
                came_from[neighbor] = current_index;
                G[neighbor] = g;
                F[neighbor] = g + heuristic(neighbor);
                if(!found(&heap, &heapLen, neighbor))
                  insert_heap(&heap, &heapLen, neighbor, &F);
              }
          }
      }

    if(pathLen > 0)                                                 //  There are no impediments, only costs, so this should never fail!
      {
        g = 0.0;
        printf("PATH:");
        for(i = 0; i < pathLen; i++)
          {
            g += C[ path[i] ];                                      //  Add up total alignment cost.
            if(i < pathLen - 1)
              printf("%d,", path[i]);
            else
              printf("%d", path[i]);
          }
        printf("|COST:%f", g);
        printf("|A:");
        for(i = 0; i < pathLen; i++)
          {
            if(i < pathLen - 1)
              printf("%d,", alignment_a[i]);
            else
              printf("%d", alignment_a[i]);
          }
        printf("|B:");
        for(i = 0; i < pathLen; i++)
          {
            if(i < pathLen - 1)
              printf("%d,", alignment_b[i]);
            else
              printf("%d", alignment_b[i]);
          }

        free(alignment_a);
        free(alignment_b);
      }

    free(C);                                                        //  Clean up. Go home.
    free(came_from);
    free(G);
    free(F);
    free(heap);
    if(pathLen > 0)
      free(path);
    return 0;
  }

/********************************************************************/

float heuristic(unsigned int index)
  {
    #ifdef __ALIGN_DEBUG
    printf("heuristic(%d)\n", index);
    #endif

    return 1.0;
  }

void reconstruct_path(unsigned int** path, unsigned int* pathLen, unsigned int** came_from, unsigned int start)
  {
    unsigned int p = 0;                                             //  Goal is always 0.
    unsigned int ctr = 1;

    #ifdef __ALIGN_DEBUG
    printf("reconstruct_path()\n");
    #endif

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

/********************************************************************/

/*  */
bool found(unsigned int** heap, unsigned int* heapLen, unsigned int index)
  {
    unsigned int i = 0;

    #ifdef __ALIGN_DEBUG
    printf("found(%d)\n", index);
    #endif

    while(i < (*heapLen) && (*heap)[i] != index)
      i++;
    return i < (*heapLen);
  }

/* Preserve heap properties, according to values in the cost matrix 'Mat'. */
void heapify(unsigned int** heap, unsigned int* heapLen, unsigned int index, float** Mat)
  {
    unsigned int left, right;
    unsigned int smallest;

    #ifdef __ALIGN_DEBUG
    printf("heapify(%d)\n", index);
    #endif

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
void insert_heap(unsigned int** heap, unsigned int* heapLen, unsigned int new_index, float** Mat)
  {
    unsigned int i = (*heapLen);

    #ifdef __ALIGN_DEBUG
    printf("insert_heap(%d)\n", new_index);
    #endif

    (*heap)[(*heapLen)] = new_index;
    (*heapLen)++;

    while(i != 0 && (*Mat)[ (*heap)[PARENT(i)] ] > (*Mat)[ (*heap)[i] ])
      {
        SWAP((*heap)[i], (*heap)[PARENT(i)]);
        i = PARENT(i);
      }

    return;
  }

/* Remove the node at index and repair the heap. */
void decrease_key(unsigned int** heap, unsigned int* heapLen, unsigned int index)
  {
    unsigned int i = index;

    #ifdef __ALIGN_DEBUG
    printf("decrease_key(%d)\n", index);
    #endif

    (*heap)[i] = UINT_MAX;                                          //  I want to take advantage of unsigned int's range,
                                                                    //  so we will treat UINT_MAX like negative infinity.
    while(i != 0)
      {
        SWAP((*heap)[i], (*heap)[PARENT(i)]);
        i = PARENT(i);
      }

    return;
  }

/*  */
unsigned int extract_min(unsigned int** heap, unsigned int* heapLen, float** Mat)
  {
    unsigned int root;

    #ifdef __ALIGN_DEBUG
    printf("extract_min()\n");
    #endif

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

/*  */
void delete_key(unsigned int** heap, unsigned int* heapLen, unsigned int index, float** Mat)
  {
    #ifdef __ALIGN_DEBUG
    printf("delete_key(%d)\n", index);
    #endif

    decrease_key(heap, heapLen, index);
    extract_min(heap, heapLen, Mat);
    return;
  }

/********************************************************************/

void print_fmatrix(unsigned int rows, unsigned int cols, float** Mat)
  {
    unsigned int i;
    for(i = 0; i < rows * cols; i++)
      {
        printf("%f ", (*Mat)[i]);
        if((i + 1) % cols == 0)
          printf("\n");
      }
    printf("\n");
    return;
  }

void print_umatrix(unsigned int rows, unsigned int cols, unsigned int** Mat)
  {
    unsigned int i;
    for(i = 0; i < rows * cols; i++)
      {
        printf("%u ", (*Mat)[i]);
        if((i + 1) % cols == 0)
          printf("\n");
      }
    printf("\n");
    return;
  }

void print_variables(unsigned int rows, unsigned int cols, float** C, float** F, float** G, unsigned int** came_from)
  {
    printf("rows = %d, cols = %d\n\n", rows, cols);
    printf("C:\n");
    print_fmatrix(rows, cols, C);
    printf("G:\n");
    print_fmatrix(rows, cols, G);
    printf("F:\n");
    print_fmatrix(rows, cols, F);
    printf("came-from:\n");
    print_umatrix(rows, cols, came_from);
    return;
  }

void print_heap(unsigned int** heap, unsigned int* heapLen, float** Mat)
  {
    unsigned int i;
    printf("Heap: ");
    for(i = 0; i < (*heapLen); i++)
      printf("[%d]:%f  ", (*heap)[i], (*Mat)[ (*heap)[i] ]);
    printf("\n");
    return;
  }