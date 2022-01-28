#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cfloat>

/* Initializes matrix with random numbers */
void initm(float *, const int, const int);

/* Pretty prints a given matrix */
void printm(const float *, const int, const int, const char *);

/* Multiplicates two matrices with O(n^3) algorithm on CPU */
void mulmat(float *, const float *, const float *, const int, const int, const int);

/* Check if number is power of 2 */
bool isPowerOfTwo(const unsigned int);

/* Pastes a sub-matrix to a bigger one */
void matcpy(float *, const float *, const int, const int, const int, const int);

/* Compare two matrices */
bool matcmp(const float *, const float *, const int);

#include "matrix.cu"

#endif