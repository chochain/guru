#include <stdio.h>
#include "c_ext.h"

#define SZ 512

__global__
void k_saxpy(int N, float a, float *x, float *y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;	// blockDim.x = number of threads/block
	if (i < N) y[i] += a*x[i];						// C = aX+B in global memory
}

__global__
void k_minit(int N, float *d_x, float *d_y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;	// for (..., i+=n_threads) ...
	if (i < N) {									// while (i<N) ...
		d_x[i] = 1.0f;
	    d_y[i] = 2.0f;
	}
}

__global__ void k_sum(int N, float *d_y) {	// sum front and back of entire array
    __shared__ float sum[SZ];				// statically allocated on device

	int t = threadIdx.x;					// in-block thread id
	int i = blockIdx.x*blockDim.x + t;		// in-grid global array index

	sum[t]= (i < N) ? d_y[i] : 0;
	__syncthreads();

	for (int s=blockDim.x>>1; s>0; s>>=1) { // binary step reducing stride width
		if (t < s) sum[t] += sum[t + s];  	// sum [t] and [t+s] into [t] i.e. stride head
		__syncthreads();
	}
	if (t==0) d_y[blockIdx.x] = sum[0]; 	// write back to each global block head
}

__forceinline__ __device__ void d_sum(int t, float *sum) {	// reduce array sum[2*SZ] into sum[0]
    for (int s=SZ; s>32; s>>=1) {			// folding by half the stride-size
       if (t < s) sum[t] += sum[t + s]; 	// add second half of the block to the first half
       __syncthreads();						// dataflow flood gate between warps
    }
    if (t<32) {								// unroll last warp, ~= 15% faster
    	sum[t]+=sum[t+32]; sum[t]+=sum[t+16]; sum[t]+=sum[t+8];
    	sum[t]+=sum[t+4];  sum[t]+=sum[t+2];  sum[t]+=sum[t+1];
    }
}

__global__ void k_sum2(int N, float *d_y) {	// sequentially executed in blocks per SM
    __shared__ float sum[2 * SZ];			// hold double the thread count

    int t = threadIdx.x;					// [0..511]
    int	i = blockIdx.x*2*SZ + t;			// global index (every 2 blocks)

    //@@ Load global array into shared memory
    sum[t]    = (i < N)    ? d_y[i]    : 0.0;	// copy first half into shared memory
    sum[SZ+t] = (SZ+i < N) ? d_y[SZ+i] : 0.0;	// copy second half

    d_sum(t, sum);							// call device function to reduce array

    //@@ Write the computed sum of the block to the block head
    if (t==0) d_y[blockIdx.x] = sum[0];
}

void echeck(const char *str) {
    cudaError err = cudaGetLastError();
    if (cudaSuccess == err) printf("\nOK> %s: ", str);
    else {
    	printf("\nERR> %s: %s\n", str, cudaGetErrorString(err));
    	exit(-1);
    }
}

void edump(int N, float msec, float *d_y) {	// write over into d_y (i.e. destructive)
//	k_sum<<<(n+SZ-1)/SZ, SZ>>>(n, d_y);		// vanilla sum
	int n    = N;
	int nblk = (n+SZ-1)/SZ/2;				// block count
	float v[2];
	do {
		k_sum2<<<nblk, SZ>>>(n, d_y);
		echeck("k_sum2()");
		cudaMemcpy(&v, d_y, sizeof(float)*2, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		printf("n=%d, nblk=%d: v[0]=%f, v[1]=%f", n, nblk, v[0], v[1]);
		n    = nblk;
		nblk = (n>SZ) ? (n+SZ-1)/SZ/2 : 1;	// last block?
	} while (n>1);

	printf("\nTotal: %f, (Bandwidth %f GB/s, %f GFLOPs)\n", v[0], N*4*3*1e-6/msec, N*2*1e-6/msec);
}

int do_cuda(void) {
  int N = 1<<24;

  float *x,   *y;
  float *d_x, *d_y;
  float *m_x, *m_y;

  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Perform SAXPY on 16M on-device elements
  cudaEventRecord(start);
  k_saxpy<<<(N+SZ-1)/SZ, SZ>>>(N, 2.0f, d_x, d_y);               // 32K blocks
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  echeck("H2D, saxpy, D2H");

  float msec = 0;
  cudaEventElapsedTime(&msec, start, stop);
  edump(N, msec, d_y);

  cudaMallocManaged(&m_x, N*sizeof(float));
  cudaMallocManaged(&m_y, N*sizeof(float));

  k_minit<<<(N+SZ-1)/SZ, SZ>>>(N, m_x, m_y);
  cudaDeviceSynchronize();
  echeck("managed mem init");

  // Perform SAXPY on 16M managed elements
  cudaEventRecord(start);
  k_saxpy<<<(N+SZ-1)/SZ, SZ>>>(N, 2.0f, m_x, m_y);               // run on host?
  echeck("managed saxpy");
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msec, start, stop);
  edump(N, msec, m_y);

  cudaFree(m_x);
  cudaFree(m_y);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}
