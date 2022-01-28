#ifndef __STRASSEN_H__
#define __STRASSEN_H__

#define BLOCK_DIM 16
#define DEPTH 10
#include "matrix.h"

float *A_dev[DEPTH], *B_dev[DEPTH], *C_dev;
float *M1[DEPTH], *M2[DEPTH], *M3[DEPTH], *M4[DEPTH], *M5[DEPTH], *M6[DEPTH], *M7[DEPTH];

bool cublas = false;

/* Sums two matrices */
void addmat(cublasHandle_t, float *, const float *, const float *, const int, const int, const int, const int, const int);

/* Subs two matrices */
void submat(cublasHandle_t, float *, const float *, const float *, const int, const int);

/* Copy matrix */
void matcpy(cublasHandle_t, float *, const float *, const int, const int, const int, const int);

/* Multiplicates two matrices with O(n^3) naive algorithm */
__global__ void matmul(float *, const float *, const float *, const int);

/* Multiplicates two matrices with Strassen's algorithm */
void strassen(cublasHandle_t, float *, const float *, const float *, const int, const int, const int);

/* Multiplicates two matrices with cuBLAS' algorithm */
void cublasMatmul(cublasHandle_t, float *, const float *, const float *, const int);

#include "strassen.cu"

#endif