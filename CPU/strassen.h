#ifndef __STRASSEN_H__
#define __STRASSEN_H__

#include "matrix.h"

/* Addizione matriciale */
void addmat(int *, const int *, const int *, const int, const int);

/* Sottrazione matriciale */
void submat(int *, const int *, const int *, const int, const int);

/* Algoritmo di Strassen */
void strassen(int *, const int *, const int *, const int);

#include "strassen.c"

#endif