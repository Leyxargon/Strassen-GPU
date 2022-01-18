#include "strassen.h"

void addmat(int *C, const int *A, const int *B, const int m, const int n) {
	int i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < m; ++j)
			C[i * m + j] = A[i * n + j] + B[i * n + j];
}

void submat(int *C, const int *A, const int *B, const int m, const int n) {
	int i, j;
	for (i = 0; i < m; ++i)
		for (j = 0; j < m; ++j)
			C[i * m + j] = A[i * n + j] - B[i * n + j];
}

void strassen(int *C, const int *A, const int *B, const int n) {
	if (n == 2) { /* caso base */
		int M1 = (A[0] + A[n + 1]) * (B[0] + B[n + 1]);	/* M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]) */
		int M2 = (A[n] + A[n + 1]) * B[0];				/* M2 = (A[1][0] + A[1][1]) * B[0][0] */
		int M3 = A[0] * (B[1] - B[n + 1]);				/* M3 = A[0][0] * (B[0][1] - B[1][1]) */
		int M4 = A[n + 1] * (B[n] - B[0]);				/* M4 = A[1][1] * (B[1][0] - B[0][0]) */
		int M5 = (A[0] + A[1]) * B[n + 1];				/* M5 = (A[0][0] + A[0][1]) * B[1][1] */
		int M6 = (A[n] - A[0]) * (B[0] + B[1]);			/* M6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1]) */
		int M7 = (A[1] - A[n + 1]) * (B[n] + B[n + 1]);	/* M7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]) */

		C[0] = M1 + M4 - M5 + M7;		/* C[0][0] */
		C[1] = M3 + M5;					/* C[0][1] */
		C[n] = M2 + M4;					/* C[1][0] */
		C[n + 1] = M1 + M3 - M2 + M6;	/* C[1][1] */
	}
	else { /* caso ricorsivo */
		/* DIVIDE */
		int m = n/2;
		
		/* matrici temporanee */
		int *a = (int *) malloc(m * m * sizeof(int));
		int *b = (int *) malloc(m * m * sizeof(int));

		/* M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]) */
		int *M1 = (int *) malloc(m * m * sizeof(int));
		addmat(a, &A[0], &A[m * (n + 1)], m, n);	/* A[0][0] + A[1][1] */
		addmat(b, &B[0], &B[m * (n + 1)], m, n);	/* B[0][0] + B[1][1] */
		strassen(M1, a, b, m);

		/* M2 = (A[1][0] + A[1][1]) * B[0][0] */
		int *M2 = (int *) malloc(m * m * sizeof(int));
		addmat(a, &A[m * n], &A[m * (n + 1)], m, n);	/* A[1][0] + A[1][1] */
		matcpy(b, &B[0], m, m, n, n);
		strassen(M2, a, b, m);
		
		/* M3 = A[0][0] * (B[0][1] - B[1][1]) */
		int *M3 = (int *) malloc(m * m * sizeof(int));
		matcpy(a, &A[0], m, m, n, n);
		submat(b, &B[m], &B[m * (n + 1)], m, n);	/* B[0][1] - B[1][1] */
		strassen(M3, a, b, m);

		/* M4 = A[1][1] * (B[1][0] - B[0][0]) */
		int *M4 = (int *) malloc(m * m * sizeof(int));
		matcpy(a, &A[m * (n + 1)], m, m, n, n);
		submat(b, &B[m * n], &B[0], m, n);	/* B[1][0] - B[0][0] */
		strassen(M4, a, b, m);

		/* M5 = (A[0][0] + A[0][1]) * B[1][1] */
		int *M5 = (int *) malloc(m * m * sizeof(int));
		addmat(a, &A[0], &A[m], m, n);	/* A[0][0] + A[0][1] */
		matcpy(b, &B[m * (n + 1)], m, m, n, n);
		strassen(M5, a, b, m);

		/* M6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1]) */
		int *M6 = (int *) malloc(m * m * sizeof(int));
		submat(a, &A[m * n], &A[0], m, n);	/* A[1][0] - A[0][0] */
		addmat(b, &B[0], &B[m], m, n);		/* B[0][0] + B[0][1] */
		strassen(M6, a, b, m);

		/* M7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]) */
		int *M7 = (int *) malloc(m * m * sizeof(int));
		submat(a, &A[m], &A[m * (n + 1)], m, n);		/* A[0][1] - A[1][1] */
		addmat(b, &B[m * n], &B[m * (n + 1)], m, n);	/* B[1][0] + B[1][1] */
		strassen(M7, a, b, m);

		/* IMPERA */

		/* C00 = M1 + M4 - M5 + M7 */
		int *C00 = (int *) malloc(m * m * sizeof(int));
		submat(a, M7, M5, m, m);	/* -(-M5 + M7) */
		addmat(b, M4, a, m, m);		/* -(-M5 + M7) + M4 */
		addmat(C00, M1, b, m, m);	/* -(-M5 + M7) + M4 + M1 */
		free(M7);

		/* C01 = M3 + M5 */
		int *C01 = (int *) malloc(m * m * sizeof(int));
		addmat(C01, M3, M5, m, m);
		free(M5);

		/* C10 = M2 + M4 */
		int *C10 = (int *) malloc(m * m * sizeof(int));
		addmat(C10, M2, M4, m, m);
		free(M4);

		/* C11 = M1 + M3 - M2 + M6 */
		int *C11 = (int *) malloc(m * m * sizeof(int));
		submat(a, M6, M2, m, m);	/* -(-M2 + M6) */
		addmat(b, M3, a, m, m);		/* -(-M2 + M6) + M3 */
		addmat(C11, M1, b, m, m);	/* -(-M2 + M6) + M3 + M1 */
		free(M6); free(M3); free(M2); free(M1);

		free(a); free(b);
		
		/* COMBINA */
		int i, j;
		for (i = 0; i < m; ++i)
			for (j = 0; j < m; ++j) {
				C[i * n + j] = C00[i * m + j];				/* C[0][0] */
				C[i * n + j + m] = C01[i * m + j];			/* C[0][1] */
				C[(i + m) * n + j] = C10[i * m + j];		/* C[1][0] */
				C[(i + m) * n + j + m] = C11[i * m + j];	/* C[1][1] */
			}

		free(C00); free(C01); free(C10); free(C11);
    }
}