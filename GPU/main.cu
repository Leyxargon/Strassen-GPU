#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <assert.h>

#include "matrix.h"
#include "strassen.h"
#include "timer.hpp"

int main(int argc, char** argv) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <matrices size> <threshold> <cublas>\n", argv[0]);
		fputs("cublas:\t0: use Strassen with naive algorithm with SM\n\t1: use Strassen with cuBLAS' algorithm\n", stderr);
		exit(0);
	}

	int n = atoi(argv[1]);
	int threshold = atoi(argv[2]);
	cublas = (bool) atoi(argv[3]);

	if (!isPowerOfTwo(n)) {
		/* rectangular matrices */
		fputs("WARNING: matrices size is not power of two\n", stdout);
		n = pow(2, ceil(log2((float) n)));
		printf("dimension of matrices = %d\n", n);
	}
	
	dim3 blocks((n + BLOCK_DIM - 1) / BLOCK_DIM, (n + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 threads(BLOCK_DIM, BLOCK_DIM);

	if (n < threshold && threshold < BLOCK_DIM) {
		fputs("ERROR: matrix size is less than threshold\n", stderr);
		exit(-2);
	}

	float *A_host = (float *) calloc(n * n, sizeof(float));
	float *B_host = (float *) calloc(n * n, sizeof(float));
	float *C_host = (float *) calloc(n * n, sizeof(float));
	float *tmp = (float *) calloc(n * n, sizeof(float));

	srand(time(NULL));
	initm(A_host, n, n);
	initm(B_host, n, n);

	printm(A_host, n, n, "A");
	printm(B_host, n, n, "B");

	/* local matrices */
	int depth, dim = n;
	for (depth = 0; depth < DEPTH && dim > 0; ++depth, dim /= 2) {
		cudaMalloc((float **) &A_dev[depth], dim * dim * sizeof(float));
		cudaMalloc((float **) &B_dev[depth], dim * dim * sizeof(float));

		if (depth == 0)
			cudaMalloc((float **) &C_dev, dim * dim * sizeof(float));
		else {
			cudaMalloc((float **) &M1[depth], dim * dim * sizeof(float));
			cudaMalloc((float **) &M2[depth], dim * dim * sizeof(float));
			cudaMalloc((float **) &M3[depth], dim * dim * sizeof(float));
			cudaMalloc((float **) &M4[depth], dim * dim * sizeof(float));
			cudaMalloc((float **) &M5[depth], dim * dim * sizeof(float));
			cudaMalloc((float **) &M6[depth], dim * dim * sizeof(float));
			cudaMalloc((float **) &M7[depth], dim * dim * sizeof(float));
		}
	}

	cudaMemcpy(A_dev[0], A_host, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev[0], B_host, n * n * sizeof(float), cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	Timer t;

	t.start();
	matmul<<<blocks, threads>>>(C_dev, A_dev[0], B_dev[0], n);
	cudaDeviceSynchronize();
	t.stop();

	printf("time for naive: %.5fms\n", t.get());
	cudaMemcpy(tmp, C_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	t.start();
	strassen(handle, C_dev, A_dev[0], B_dev[0], n, 0, threshold);
	t.stop();

	printf("time for strassen: %.5fms\n", t.get());
	cudaMemcpy(C_host, C_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	if (cublas) {
		t.start();
		cublasMatmul(handle, C_dev, A_dev[0], B_dev[0], n);
		t.stop();
		
		printf("time for cublas: %.5fms\n", t.get());
		cudaMemcpy(C_host, C_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);
	}

	cublasDestroy(handle);

	int i;
	for (i = 0; i < depth; ++i) {
		cudaFree(A_dev[i]); cudaFree(B_dev[i]);

		if (i == 0)
			cudaFree(C_dev);
		else {
			cudaFree(M1[i]);
			cudaFree(M2[i]);
			cudaFree(M3[i]);
			cudaFree(M4[i]);
			cudaFree(M5[i]);
			cudaFree(M6[i]);
			cudaFree(M7[i]);
		}
	}

	printm(C_host, n, n, "C");
	printf(matcmp(tmp, C_host, n) ? "Strassen OK\n" : "Strassen not working\n");

	cudaDeviceReset();

	free(A_host); free(B_host); free(C_host); free(tmp);

	return 0;
}