#include "matrix.h"

void initm(float *M, const int m, const int n) {
	int i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < n; ++j)
			M[i * n + j] = rand() % 10;
}

void printm(const float *M, const int m, const int n, const char *str) {
	if (m > 32 || n > 32)
		return;

	int i, j;
	printf("%s =\n{", str);
	for (i = 0; i < m; ++i) {
		printf(i == 0 ? "[ " : " [ ");
		for (j = 0; j < n; ++j)
			printf(M[i * n + j] <= 9 ? "  %.0f " : (M[i * n + j] <= 99 ? " %.0f " : "%.0f "), M[i * n + j]);
		printf(i < m - 1 ? "]\n" : "]");
	}
	printf("}\n");
}

void mulmat(float *C, const float *A, const float *B, const int m, const int n, const int p) {
	int i, j, k;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < p; ++j) {
			C[i * p + j] = 0;

			for (k = 0; k < n; ++k)
				C[i * p + j] += A[i * n + k] * B[k * p + j];
		}
	}
}

bool isPowerOfTwo(const unsigned int n) {
	return (n != 0) && ((n & (n - 1)) == 0);
}

bool matcmp(const float *A, const float *B, const int n) {
	int i, j;
	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j)
			if (abs(A[i * n + j] - B[i * n + j]) >= FLT_EPSILON)
				return false;
	
	return true;
}