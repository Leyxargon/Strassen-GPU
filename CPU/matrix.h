#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <time.h>
#include <math.h>

/* Inizializza matrici con numeri casuali */
void initm(int *, const int, const int);

/* Incolla una sotto-matrice in una matrice piu grande */
void matcpy(int *, const int *, const int, const int, const int, const int);

/* Verifica se due matrici sono uguali */
bool matcmp(const int *, const int *, const int, const int);

/* Stampa una matrice con output formattati */
void printm(const int *, const int, const int, const char *);

/* Prodotto matriciale O(n^3) */
void mulmat(int *, const int *, const int *, const int, const int, const int);

/* Ricerca il numero massimo */
int max(unsigned int, ...);

/* Verifica se il numero è una potenza di due */
bool isPowerOfTwo(const unsigned int);

#include "matrix.c"

#endif