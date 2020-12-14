#include<stdio.h>
#include "dehaze_kernel.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#define MAX_SIZE 3000
#define MAX_VALUE 255.0
#define TILE_DIM 16

extern __global__ void transpose(float *odata, float *idata, int width, int height);

__global__ void d_min_img_kernel(float *src, float *dst, int iWidth, int iHeight, int iChannel){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float r, g, b;
	if (id<iHeight*iWidth){
		r = src[id*iChannel + 0];
		g = src[id*iChannel + 1];
		b = src[id*iChannel + 2];
		dst[id] = (r<g) ? r : g;
		dst[id] = (dst[id]<b) ? dst[id] : b;
	}
}


__global__ void d_minfilter_x(float *src, float *dst, int width, int height, int r, int BLOCKSIZE){
	int i, j;
	int mask, len, extra, num, head, rear;
	int bid, tid;
	bid = blockIdx.x;
	tid = threadIdx.x;
	//float *p, *q;
	__shared__ float g[MAX_SIZE]; //跟图像长宽大小有关，图像过大要注意调整这里的大小！！！
	__shared__ float h[MAX_SIZE];

	mask = 2 * r + 1;
	len = width + 2 * r + mask - (width + 2 * r) % mask;  //补齐之后的长度
	extra = len - width - r;//图像右边多出来的部分
	num = len / mask;  //共num 个滤波核


	if (bid<height){
		//p = src + bid * width; // 行首地址
		//q = dst + bid * width;

		for (i = tid; i<r; i += BLOCKSIZE){
			g[i] = MAX_VALUE;
			h[i] = MAX_VALUE;
		}
	__syncthreads();
		for (i = tid; i<width; i += BLOCKSIZE){
			g[r + i] = src[bid * width + i];
			h[r + i] = src[bid * width + i];
		}
	__syncthreads();
		for (i = tid; i<extra; i += BLOCKSIZE){
			g[r + width + i] = MAX_VALUE;
			h[r + width + i] = MAX_VALUE;
		}//补齐

		__syncthreads();
		for (i = tid; i<num; i += BLOCKSIZE){
			head = i*mask;
			rear = (i + 1)*mask - 1;
			for (j = head + 1; j<(head + mask); j++){
				g[j] = (g[j - 1] < g[j]) ? g[j - 1] : g[j];
				h[rear - j + head] = (h[rear - j + head + 1] < h[rear - j + head]) ? h[rear - j + head + 1] : h[rear - j + head];
			}
		}//计算g，h
		__syncthreads();
		for (i = tid; i<width; i += BLOCKSIZE)
			dst[bid * width + i] = (g[i + r + r] > h[i]) ? h[i] : g[i + r + r];
		//dst[bid * width + i] = g[i+r];
	}
}

__global__ void d_minfilter_y(float *src, float *dst, int width, int height, int r, int BLOCKSIZE){
	int i, j;
	int mask, len, extra, num, head, rear;
	int bid, tid;
	bid = blockIdx.x;
	tid = threadIdx.x;
	//float *p, *q;

	extern __shared__ float g[ ];
	extern __shared__ float h[ ];

	mask = 2 * r + 1;
	len = height + 2 * r + mask - (height + 2 * r) % mask;
	extra = len - height - r;
	num = len / mask;


	if (bid<width){
		//p = src + bid;
		//q = dst + bid;
		for (i = tid; i<height; i += BLOCKSIZE){
			g[r + i] = src[i*width + bid];
			h[r + i] = src[i*width + bid];
		}

		for (i = tid; i<r; i += BLOCKSIZE){
			g[i] = MAX_VALUE;
			h[i] = MAX_VALUE;
		}


		for (i = tid; i<extra; i += BLOCKSIZE){
			g[r + height + i - 1] = MAX_VALUE;
			h[r + height + i - 1] = MAX_VALUE;
		}//补齐

		for (i = tid; i<num; i += BLOCKSIZE){
			head = i*mask;
			rear = (i + 1)*mask - 1;
			for (j = head + 1; j<head + mask; j++){
				g[j] = (g[j - 1] < g[j]) ? g[j - 1] : g[j];
				h[rear - j + head] = (h[rear - j + head + 1] < h[rear - j + head]) ? h[rear - j + head + 1] : h[rear - j + head];
			}
		}//计算g，h
		__syncthreads();
		for (i = tid; i<height; i += BLOCKSIZE)
			dst[i*width + bid] = (g[i + r + r] > h[i]) ? h[i] : g[i + r + r];

	}
}

extern "C"
void minfilter(float *d_fog, float *d_min_img, float *d_win_dark, float *d_temp, int width, int height, int channel, int radius, int BLOCKSIZE){

	dim3 grid1(width / TILE_DIM + 1, height / TILE_DIM + 1);
	dim3 grid2(height / TILE_DIM + 1, width / TILE_DIM + 1);
	dim3 block(TILE_DIM, TILE_DIM);


	if (channel == 3)
	{
		d_min_img_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_fog, d_min_img, width, height, channel);
	}
	
	d_minfilter_x << <height, BLOCKSIZE>> >(d_min_img, d_temp, width, height, radius, BLOCKSIZE);
	transpose << <grid1, block >> >(d_min_img, d_temp, width, height);
	d_minfilter_x << <width, BLOCKSIZE >> >(d_min_img, d_temp, height, width, radius, BLOCKSIZE);
	transpose << <grid2, block >> >(d_win_dark, d_temp, height, width);

	//d_minfilter_x<<<iWidth,BLOCKSIZE, len1>>>(d_temp1, d_temp2, iHeight, iWidth, radius, BLOCKSIZE               );
	//d_minfilter_y<<<iWidth,BLOCKSIZE, len2>>>(d_temp, d_win_dark, iWidth, iHeight, radius, BLOCKSIZE               );

}