#include "matrix.h"

void initm(int *M, const int m, const int n) {
	int i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < n; ++j)
			M[i * n + j] = rand()%12;
}

void matcpy(int *dst, const int *src, const int m_dst, const int n_dst, const int m_src, const int n_src) {
	int i, j;
	if (m_dst < m_src || n_dst < n_src)
		for (i = 0; i < m_dst; ++i)
			for (j = 0; j < n_dst; ++j)
				dst[i * n_dst + j] = src[i * n_src + j];
	else
		for (i = 0; i < m_src; ++i)
			for (j = 0; j < n_src; ++j)
				dst[i * n_dst + j] = src[i * n_src + j];
}

bool matcmp(const int *A, const int *B, const int m, const int n) {
	int i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < n; ++j)
			if (A[i * n + j] != B[i * n + j])
				return false;
	
	return true;
}

void printm(const int *M, const int m, const int n, const char *str) {
	if (m > 32 || n > 32)
		return;

	int i, j;
	printf("%s =\n{", str);
	for (i = 0; i < m; ++i) {
		printf(i == 0 ? "[ " : " [ ");
		for (j = 0; j < n; ++j)
			printf(M[i * n + j] <= 9 ? "  %d " : (M[i * n + j] <= 99 ? " %d " : "%d "), M[i * n + j]);
		printf(i < m - 1 ? "]\n" : "]");
	}
	printf("}\n");
}

void mulmat(int *C, const int *A, const int *B, const int m, const int n, const int p) {
	int i, j, k;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < p; ++j) {
			C[i * p + j] = 0;

			for (k = 0; k < n; ++k)
				C[i * p + j] += A[i * n + k] * B[k * p + j];
		}
	}
}

int max(unsigned int numArgs, ...) {
	va_list numbers;
    va_start(numbers, numArgs);

    int max = va_arg(numbers, int);
	int i = 0, tmp;

    while (++i < numArgs)
		if ((tmp = va_arg(numbers, int)) > max)
			max = tmp;

    va_end(numbers);
    return max;
}

bool isPowerOfTwo(const unsigned int n) {
    return (n != 0) && ((n & (n - 1)) == 0);
}