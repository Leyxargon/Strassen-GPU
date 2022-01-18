#include "matrix.h"
#include "strassen.h"

int main(int argc, char **argv) {
	if (argc != 4) {
		fprintf(stderr, "uso: %s <m> <n> <p>\n", argv[0]);
		fprintf(stderr, "dim:\tm = righe della matrice A\n\tn = colonne della matrice A e righe della matrice B\n\tp = colonne della matrice B\n");
		exit(-1);
	}

	srand(time(NULL));

	int m, n, p;

	if ((m = atoi(argv[1])) < 2 || (n = atoi(argv[2])) < 2 || (p = atoi(argv[3])) < 2) {
		fprintf(stderr, "uso: %s <m> <n> <p>\n", argv[0]);
		fprintf(stderr, "dim:\tm = righe della matrice A\n\tn = colonne della matrice A e righe della matrice B\n\tp = colonne della matrice B\n");
		exit(-2);
	}

	int *A = (int *) malloc(m * n * sizeof(int));
	int *B = (int *) malloc(n * p * sizeof(int));
	int *C = (int *) calloc(m * p, sizeof(int));

	initm(A, m, n);
	initm(B, n, p);

	if (m == n && m == p && isPowerOfTwo(m) && isPowerOfTwo(n) && isPowerOfTwo(p)) {
		clock_t t = clock();
		strassen(C, A, B, n);
		t = clock() - t;
		printf("tempo per strassen(): %f seconds\n", ((double) t) / CLOCKS_PER_SEC);
	}
	else {
		int q = pow(2, ceil(log2(max(3, m, n, p))));

		int *a = (int *) calloc(q * q, sizeof(int));
		int *b = (int *) calloc(q * q, sizeof(int));
		int *c = (int *) calloc(q * q, sizeof(int));

		matcpy(a, A, q, q, m, n);
		matcpy(b, B, q, q, n, p);

		clock_t t = clock();
		strassen(c, a, b, q);
		t = clock() - t;
		printf("tempo per strassen(): %f seconds\n", ((double) t) / CLOCKS_PER_SEC);

		matcpy(C, c, m, p, q, q);

		printm(a, q, q, "a");
		printm(b, q, q, "b");
		printm(c, q, q, "c");

		free(a); free(b); free(c);
	}
	printm(A, m, n, "A");
	printm(B, n, p, "B");
	printm(C, m, p, "C");

	int *T = (int *) malloc(m * p * sizeof(int));
	clock_t t = clock();
	mulmat(T, A, B, m, n, p);
	t = clock() - t;
	printf("tempo per mulmat(): %f seconds\n", ((double) t) / CLOCKS_PER_SEC);
	
	printf(matcmp(T, C, m, p) ? "Strassen OK\n" : "Strassen NON FUNZIONA\n");

	free(A); free(B); free(C); free(T);
	return 0;
}