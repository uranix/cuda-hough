#include "Hough.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace Magick;

#define CUDA_CHECK(x) do { \
	cudaError_t __err = (x); \
	if (__err != cudaSuccess) { \
		std::cerr << "CUDA error at line " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(__err) << std::endl; \
	} } while (0)

#define BLOCK_SIZE 512
#define TILE_DIM 16

__device__ void hough_line(int w, int h, int *img, int lda, float alpha, int *res, bool fwd) {
	/* alpha in [-pi/2, pi/2) */
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int n = w + h - 1;

	if (tid < n) {

		float xc = .5f * (1 - h);
		float yc = -xc;
		float ta = tanf(alpha);

		int x0 = static_cast<int>(xc + yc * ta + .5f);
		int x1 = 1 - h - x0;

		x0 += tid;
		x1 += tid;

		int dx = abs(x1 - x0);
		int dy = h - 1;
		
		int err = 0;
		int x = x0;
		int xstep = (x1 > x0) ? 1 : -1;

		int sum = 0;

		for (int y = 0; y < h; y++) {

			if (x >= 0 && x < w) {
				sum += img[y * lda + x];
			}

			err += dx;
			if (2 * err >= dy) {
				x += xstep;
				err -= dy;
			}
		}

		if (fwd)
			res[tid] = sum;
		else
			res[n - tid] = sum;
	}
}

__global__ void hough_transform_kernel(int width, int height, int *image, int lda,
	int anglelo, int anglehi, int angles, int *res, bool fwd)
{
	int n = height + width - 1;

	const float pi = 3.14159265358979f;

	for (int ia = anglelo; ia < anglehi; ia++) {
		float alpha = ia * pi / (2 * angles) - pi / 4;
		hough_line(width, height, image, lda, alpha, res + (fwd ? ia : angles - ia) * n, fwd);
	}
}

__global__ void copy_aligned_n(int *dalign, const PixelPacket *img, int width, int height) {
	int x = threadIdx.x + blockIdx.x * TILE_DIM;
	int y = threadIdx.y + blockIdx.y * TILE_DIM;

	int walign = TILE_DIM * gridDim.x;

	if (x < width && y < height)
		dalign[y * walign + x] = img[y * width + x].green;
}

__global__ void copy_aligned_t(int *dalign, const PixelPacket *img, int width, int height) {
	int bx = blockIdx.x * TILE_DIM;
	int by = blockIdx.y * TILE_DIM;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ int smem[TILE_DIM + 1][TILE_DIM + 1];

	int halign = TILE_DIM * gridDim.y;

	if ((tx + bx) < width && (ty + by) < height)
		smem[tx][ty] = img[(by + ty) * width + bx + tx].green;
	
	__syncthreads();

	dalign[(bx + ty) * halign + by + tx] = smem[ty][tx];
}

void hough_transform(int width, int height, const PixelPacket *image, int angles, int *res) {
	int n = height + width - 1;
	int bs = BLOCK_SIZE;
	int astep = 32;	

	dim3 block(bs);
	dim3 grid((n + bs - 1) / bs);

	int walign = (width + TILE_DIM - 1) & ~(TILE_DIM - 1);
	int halign = (height + TILE_DIM - 1) & ~(TILE_DIM - 1);

	PixelPacket *dimg;
	int *dres;
	int *daligned;

	CUDA_CHECK(cudaMalloc((void **)&dimg, sizeof(PixelPacket) * height * width));
	CUDA_CHECK(cudaMalloc((void **)&dres, sizeof(int) * angles * n * 2));	
	CUDA_CHECK(cudaMalloc((void **)&daligned, sizeof(int) * walign * halign));

	CUDA_CHECK(cudaMemcpy(dimg, image, sizeof(PixelPacket) * height * width, cudaMemcpyHostToDevice));

	dim3 copyblock(TILE_DIM, TILE_DIM);
	dim3 copygrid(walign / TILE_DIM, halign / TILE_DIM);

	copy_aligned_n<<<copygrid, copyblock>>>(daligned, dimg, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	for (int alo = 0; alo < angles; alo += astep)
		hough_transform_kernel<<<grid, block>>>(width, height, daligned, walign,
			alo, std::min(alo + astep, angles), angles, dres, false);

	copy_aligned_t<<<copygrid, copyblock>>>(daligned, dimg, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	for (int alo = 0; alo < angles; alo += astep)
		hough_transform_kernel<<<grid, block>>>(height, width, daligned, halign,
			alo, std::min(alo + astep, angles), angles, dres + angles * n, true);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(res, dres, sizeof(int) * angles * n * 2, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dimg));
	CUDA_CHECK(cudaFree(dres));
	CUDA_CHECK(cudaFree(daligned));
}
