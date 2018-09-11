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

__global__ void k_sum(int N, float *d_y) {	// sum front and back of entire array, destructive
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

__global__ void k_sum2(int N, float *o_y, float *i_y) {	// sequentially executed in blocks per SM
    __shared__ float sum[2 * SZ];			// hold double the thread count

    int t = threadIdx.x;					// [0..511]
    int	i = blockIdx.x*2*SZ + t;			// global index (every 2 blocks)

    sum[t]    = (i < N)    ? i_y[i]    : 0.0;	// copy first half into shared memory
    sum[SZ+t] = (SZ+i < N) ? i_y[SZ+i] : 0.0;	// copy second half

    d_sum(t, sum);							// call device function to reduce array

    if (t==0) o_y[blockIdx.x] = sum[0];		// put sum into block head
}

void echeck(const char *str) {
	cudaDeviceSynchronize();
    cudaError err = cudaGetLastError();
    if (cudaSuccess == err) printf("\nOK> %s: ", str);
    else {
    	printf("\nERR> %s: %s\n", str, cudaGetErrorString(err));
    	exit(-1);
    }
}

void bmark(int N, float msec, float *d_y) {	// dump array sum
//	k_sum<<<(N+SZ-1)/SZ, SZ>>>(N, d_y);		// vanilla sum, SZ threads/block
	int n = N, SZ2 = SZ*2;
	int nblk = (n+SZ2-1)/SZ2;				// double-width block count

	float v[2];
	float *o_y, *i_y = d_y;
	cudaMalloc(&o_y, (nblk>1 ? nblk : 2)*sizeof(float));	// allocate output array, sync here
	echeck("bmark malloc");

	do {									// recursively down to 1 value
		k_sum2<<<nblk, SZ>>>(n, o_y, i_y);
		echeck("k_sum2()");
		cudaMemcpy(&v, o_y, sizeof(float)*2, cudaMemcpyDeviceToHost);	// warp sync here
		printf("n=%d, nblk=%d: v[0]=%f, v[1]=%f", n, nblk, v[0], v[1]);
		n    = nblk;
		nblk = (n+SZ2-1)/SZ2;				// reduce by 2*block count
		i_y  = o_y;							// point to output array
	} while (n>1);

	printf("\nTotal: %f, (Bandwidth %f GB/s, %f GFLOPs)\n", v[0], N*4*3*1e-6/msec, N*2*1e-6/msec);
	cudaFree(o_y);							// release, async
}

int do_cuda(void) {
  int N = (1<<24)+1;

  float *x, *y, *d_x, *d_y, *m_x, *m_y;

  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i=0; i<N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  // Perform SAXPY on 16M on-device elements
  cudaEventRecord(ev0);
  k_saxpy<<<(N+SZ-1)/SZ, SZ>>>(N, 2.0f, d_x, d_y);               // 32K blocks
  cudaEventRecord(ev1);
  echeck("H2D, saxpy, D2H");

  float msec = 0;
  cudaEventElapsedTime(&msec, ev0, ev1);
  bmark(N, msec, d_y);											// sync included

  cudaMallocManaged(&m_x, N*sizeof(float));
  cudaMallocManaged(&m_y, N*sizeof(float));

  k_minit<<<(N+SZ-1)/SZ, SZ>>>(N, m_x, m_y);
  echeck("managed mem init");

  // Perform SAXPY on 16M managed elements
  cudaEventRecord(ev0);
  k_saxpy<<<(N+SZ-1)/SZ, SZ>>>(N, 2.0f, m_x, m_y);              // run on host?
  cudaEventRecord(ev1);
  echeck("managed saxpy");
  cudaEventElapsedTime(&msec, ev0, ev1);
  bmark(N, msec, m_y);											// sync included

  cudaEventDestroy(ev0);										// release event objects
  cudaEventDestroy(ev1);
  cudaFree(m_x);
  cudaFree(m_y);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}
