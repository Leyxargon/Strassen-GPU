#include "strassen.h"

__global__ void matmul(float *C, const float *A, const float *B, const int n) {
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int nn = gridDim.x;

	__shared__ float sharedA[BLOCK_DIM][BLOCK_DIM], sharedB[BLOCK_DIM][BLOCK_DIM];

	if (i < n && j < n) {
		float sum = 0;
		for (int k = 0; k < nn; ++k) {
			sharedA[threadIdx.y][threadIdx.x] = A[i * n + k * BLOCK_DIM + threadIdx.x];
			sharedB[threadIdx.y][threadIdx.x] = B[(k * BLOCK_DIM + threadIdx.y) * n + j];
			__syncthreads();

			for (int l = 0; l < BLOCK_DIM; ++l)
				sum += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
			__syncthreads();
		}

		C[i * n + j] = sum;
	}
}

void cublasMatmul(cublasHandle_t handle, float *C, const float *A, const float *B, const int dim) {
	const float alpha = 1.0;
	const float beta = 0.0;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, B, dim, A, dim, &beta, C, dim);
}

void addmat(cublasHandle_t handle, float *C, const float *A, const float *B, const int m, const int n, const int lda, const int ldb, const int ldc) {
	const float plus = 1.0;
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &plus, A, lda, &plus, B, ldb, C, ldc);
}

void submat(cublasHandle_t handle, float *C, const float *A, const float *B, const int m, const int n, const int lda, const int ldb, const int ldc) {
	const float minus = -1.0;
	const float plus = 1.0;
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &plus, A, lda, &minus, B, ldb, C, ldc);
}

void matcpy(cublasHandle_t handle, float *B, const float *A, const int m, const int n, const int lda, const int ldb) {
	const float zero = 0.0;
	const float plus = 1.0;
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &plus, A, lda, &zero, B, ldb, B, ldb);
}

void strassen(cublasHandle_t handle, float *C, const float *A, const float *B, const int n, const int d, const int threshold) {
	const int m = n / 2;

	if (n <= threshold) {
		if (cublas)
			cublasMatmul(handle, C, A, B, n);
		else {
			dim3 blocks((n + BLOCK_DIM - 1) / BLOCK_DIM, (n + BLOCK_DIM - 1) / BLOCK_DIM);
			dim3 threads(BLOCK_DIM, BLOCK_DIM);
			matmul<<<blocks, threads>>>(C, A, B, n);
		}
	}
	else {
		/* M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]) */
		addmat(handle, A_dev[d + 1], &A[0], &A[m * n + m], m, m, n, n, m);	/* A[0][0] + A[1][1] */
		addmat(handle, B_dev[d + 1], &B[0], &B[m * n + m], m, m, n, n, m);	/* B[0][0] + B[1][1] */
		strassen(handle, M1[d + 1], A_dev[d + 1], B_dev[d + 1], m, d + 1, threshold);

		/* M2 = (A[1][0] + A[1][1]) * B[0][0] */
		addmat(handle, A_dev[d + 1], &A[m * n], &A[m * n + m], m, m, n, n, m);	/* A[1][0] + A[1][1] */
		matcpy(handle, B_dev[d + 1], &B[0], m, m, n, m);
		strassen(handle, M2[d + 1], A_dev[d + 1], B_dev[d + 1], m, d + 1, threshold);

		/* M3 = A[0][0] * (B[0][1] - B[1][1]) */
		matcpy(handle, A_dev[d + 1], &A[0], m, m, n, m);
		submat(handle, B_dev[d + 1], &B[m], &B[m * n + m], m, m, n, n, m);	/* B[0][1] - B[1][1] */
		strassen(handle, M3[d + 1], A_dev[d + 1], B_dev[d + 1], m, d + 1, threshold);

		/* M4 = A[1][1] * (B[1][0] - B[0][0]) */
		matcpy(handle, A_dev[d + 1], &A[m * n + m], m, m, n, m);
		submat(handle, B_dev[d + 1], &B[m * n], &B[0], m, m, n, n, m);	/* B[1][0] - B[0][0] */
		strassen(handle, M4[d + 1], A_dev[d + 1], B_dev[d + 1], m, d + 1, threshold);

		/* M5 = (A[0][0] + A[0][1]) * B[1][1] */
		addmat(handle, A_dev[d + 1], &A[0], &A[m], m, m, n, n, m);	/* A[0][0] + A[0][1] */
		matcpy(handle, B_dev[d + 1], &B[m * n + m], m, m, n, m);
		strassen(handle, M5[d + 1], A_dev[d + 1], B_dev[d + 1], m, d + 1, threshold);

		/* M6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1]) */
		submat(handle, A_dev[d + 1], &A[m * n], &A[0], m, m, n, n, m);	/* A[1][0] - A[0][0] */
		addmat(handle, B_dev[d + 1], &B[0], &B[m], m, m, n, n, m);		/* B[0][0] + B[0][1] */
		strassen(handle, M6[d + 1], A_dev[d + 1], B_dev[d + 1], m, d + 1, threshold);

		/* M7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]) */
		submat(handle, A_dev[d + 1], &A[m], &A[m * n + m], m, m, n, n, m);		/* A[0][1] - A[1][1] */
		addmat(handle, B_dev[d + 1], &B[m * n], &B[m * n + m], m, m, n, n, m);	/* B[1][0] + B[1][1] */
		strassen(handle, M7[d + 1], A_dev[d + 1], B_dev[d + 1], m, d + 1, threshold);


		/* C00 = M1 + M4 - M5 + M7 */
		matcpy(handle, &C[0], M1[d + 1], m, m, m, n);
		addmat(handle, &C[0], &C[0], M4[d + 1], m, m, n, m, n);
		submat(handle, &C[0], &C[0], M5[d + 1], m, m, n, m, n);
		addmat(handle, &C[0], &C[0], M7[d + 1], m, m, n, m, n);

		/* C01 = M3 + M5 */
		matcpy(handle, &C[m], M3[d + 1], m, m, m, n);
		addmat(handle, &C[m], &C[m], M5[d + 1], m, m, n, m, n);

		/* C10 = M2 + M4 */
		matcpy(handle, &C[m * n], M2[d + 1], m, m, m, n);
		addmat(handle, &C[m * n], &C[m * n], M4[d + 1], m, m, n, m, n);

		/* C11 = M1 + M3 - M2 + M6 */
		matcpy(handle, &C[m * n + m], M1[d + 1], m, m, m, n);
		submat(handle, &C[m * n + m], &C[m * n + m], M2[d + 1], m, m, n, m, n);
		addmat(handle, &C[m * n + m], &C[m * n + m], M3[d + 1], m, m, n, m, n);
		addmat(handle, &C[m * n + m], &C[m * n + m], M6[d + 1], m, m, n, m, n);
	}
}