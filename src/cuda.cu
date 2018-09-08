#include <stdio.h>
#include "c_ext.h"

__global__
void saxpy(int N, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N) y[i] = a*x[i] + y[i];
}

__global__
void d_m_init(int N, float *d_x, float *d_y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N) {
		d_x[i] = 1.0f;
	    d_y[i] = 2.0f;
	}
}

void edump(int N, float msec, float *y)
{
	float dif = 0.0f;
	for (int i = 0; i < N; i++)
		dif = max(dif, abs(y[i] - 4.0f));
	printf("Max delta: %f, (Bandwidth %f GB/s, %f GFLOPs)\n", dif, N*4*3*1e-6/msec, N*2*1e-6/msec);
}

int do_cuda(void)
{
  int N = 1<<24;
  float *x, *y, *d_x, *d_y;
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
  saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);               // async
  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);  // continue running
  cudaEventSynchronize(stop);
  float msec = 0;
  cudaEventElapsedTime(&msec, start, stop);
  edump(N, msec, y);

  cudaMallocManaged(&m_x, N*sizeof(float));
  cudaMallocManaged(&m_y, N*sizeof(float));

  d_m_init<<<(N+511)/512, 512>>>(N, m_x, m_y);

  // Perform SAXPY on 16M managed elements
  cudaEventRecord(start);
  saxpy<<<(N+511)/512, 512>>>(N, 2.0f, m_x, m_y);               // run on host?
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
