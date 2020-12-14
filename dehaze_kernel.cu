#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#define MIN_VALUE -1
#define BLOCK_ROWS 16
#define TILE_DIM 16

__device__ float mymin(float a, float b)
{
	return a<b ? a : b;
}
__device__ float mymax(float a, float b)
{
	return a>b ? a : b;
}

__device__ unsigned char round_me(float x)
{
	if (x>255) return 255;
	else if (x<0) return 0;
	else return x;
}

__global__ void float_fog_kernel(unsigned char *src, float *dst, float *d_foggy_gray, int width, int height, int channel){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < height*width)
	{
		for (int k = 0; k < channel; ++k)
		{
			dst[id*channel + k] = 1.0 * src[id*channel + k];
		}
		if (channel == 3)
			d_foggy_gray[id] = src[(id)*channel + 0] * 0.2989f + src[(id)*channel + 1] * 0.5870f + src[(id)*channel + 2] * 0.1140f;
		else
			d_foggy_gray[id] = src[id] * 1.0;
	}
}

//dst=src1+src2
__global__ void matrix_add(float *src1, float *src2, float *dst, int row, int column){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column){
		dst[id] = src1[id] + src2[id];
	}
}

//dst=src1-src2
__global__ void matrix_subtract(float *src1, float *src2, float *dst, int row, int column){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column)//
	{
		dst[id] = src1[id] - src2[id];
	}
}

//dst=src+sum
__global__ void matrix_add_all(float *src, float *dst, int row, int column, float sum)
{
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column)//
	{
		dst[id] = src[id] + sum;
	}
}

//dst=src./div
__global__ void matrix_divide(float *src, float *dst, int row, int column, float div){

	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column)//
	{
		dst[id] = src[id] / div;
	}
}

//dot divide  dst=src1./src2
__global__ void matrix_dot_divide(float *src1, float *src2, float *dst, int row, int column){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column)//
	{
		dst[id] = src1[id] / src2[id];
	}
}

//dot multiple  dst=src1.*src2
__global__ void matrix_dot_multiple(float *src1, float *src2, float *dst, int row, int column){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column)//
	{
		dst[id] = src1[id] * src2[id];
	}

}

__global__ void matrix_cal_a(float *mean_I, float *mean_p, float *mean_II, float *mean_Ip, float *a, float eps, int row, int column){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column)//
	{
		a[id] = (mean_Ip[id] - mean_I[id] * mean_p[id]) / (mean_II[id] - mean_I[id] * mean_I[id] + eps);
	}
}

__global__ void matrix_cal_b(float *mean_I, float *mean_p, float *mean_II, float *mean_Ip, float *b, float eps, int row, int column){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id<row*column)
	{
		b[id] = mean_p[id] - mean_I[id] * (mean_Ip[id] - mean_I[id] * mean_p[id]) / (mean_II[id] - mean_I[id] * mean_I[id] + eps);
	}
}

__global__ void matrix_cal_q(float *mean_a, float *mean_b, float *I, float *q, int width, int height){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;
	if (id < width*height)
	{
		q[id] = mean_a[id] * I[id] + mean_b[id];
	}
}



//在minfilter和bofilter中都会使用到的优化方法
__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	if (xIndex<width && yIndex<height){
		int index_in = xIndex + (yIndex)*width;
		tile[threadIdx.y][threadIdx.x] = idata[index_in];
	}
	__syncthreads();

	int xIndex_new = blockIdx.y * TILE_DIM + threadIdx.x;
	int yIndex_new = blockIdx.x * TILE_DIM + threadIdx.y;
	if (xIndex_new<height && yIndex_new<width){
		//for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
		{
			int index_out = xIndex_new + (yIndex_new)*height;
			//odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
			odata[index_out] = tile[threadIdx.x][threadIdx.y];
		}
	}
}

//height*width/1024,512 取输入雾图暗通道图的0.1%最亮的点
__global__ void MaxReductionkernel(float *d_win_dark, float *d_im_dark, int width, int height, int *d_index)
{
	__shared__ float win_dark[512];
	__shared__ int index[512];

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx;
	float data1, data2;
	int iDim;

	idx = bid * 1024 + tid;//bid = width*height*iEndNumber/512

	if (bid < height*width / 1024 + 1)//Initialize
	{
		win_dark[tid] = MIN_VALUE;
		index[tid] = 0;
		//__syncthreads();
	}

	if (idx <height*width)//Determine first to compute 512 pixels
	{
		//index[tid] = 0;
		data1 = d_win_dark[idx];

		if (idx + 512 <height*width)
			data2 = d_win_dark[idx + 512];
		else
			data2 = 0;

		if (data1 < data2)
		{
			win_dark[tid] = data2;//initialize index
			index[tid] = idx + 512;
		}
		else
		{
			win_dark[tid] = data1;//initialize index
			index[tid] = idx;
		}
	}
	__syncthreads();

	//find max value in all max values
	for (iDim = 256; iDim>0; iDim = iDim / 2)
	{
		if (tid<iDim)
		{
			if (win_dark[tid]<win_dark[tid + iDim])//find and swap max value to first half
			{
				win_dark[tid] = win_dark[tid + iDim];
				index[tid] = index[tid + iDim];
			}
		}
		__syncthreads();
	}

	if (tid == 0)  //判断这里必须加上bid<height*width/1024，因为分了bid<height*width/1024+1个块但之前的计算如果height*width/1024为整，最后一个快是没有算的，为空
	{
		d_im_dark[bid] = win_dark[0];
		d_index[bid] = index[0];
	}
	//Need to remember the correct index

}

//height*width/1024/512,256 取暗通道图最亮的0.1%的点对应雾图中最亮的点
__global__ void MaxReductionkernelTwo(float *d_im_dark, float *d_win_dark, float *d_foggy_gray, int width, int height, int *d_index)
{
	__shared__ float win_dark[256];
	__shared__ int index[256];

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx;
	int location;
	float data1, data2;
	int iDim;

	idx = bid * 512 + tid;//bid = width*height*iEndNumber/512
	if (bid<height*width / 1024 / 512 + 1)//Initialize
	{
		win_dark[tid] = MIN_VALUE;
		index[tid] = 0;
	}
	__syncthreads();
	if (idx < height*width / 1024)//Determine first to compute 512 pixels
	{
		index[tid] = 0;
		location = d_index[idx];
		data1 = d_foggy_gray[location];
		if (idx + 256 < height*width / 1024){
			location = d_index[idx + 256];
			data2 = d_foggy_gray[location];
		}
		else
			data2 = 0;

		if (data1 < data2)
		{
			win_dark[tid] = data2;//initialize index
			index[tid] = d_index[idx + 256];
		}
		else
		{
			win_dark[tid] = data1;//initialize index
			index[tid] = d_index[idx];
		}
	}
	__syncthreads();

	//find max value in all max values
	for (iDim = 128; iDim>0; iDim = iDim / 2)
	{
		if (tid<iDim)
		{
			if (win_dark[tid]<win_dark[tid + iDim])//find and swap max value to first half
			{
				win_dark[tid] = win_dark[tid + iDim];
				index[tid] = index[tid + iDim];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		d_win_dark[bid] = win_dark[0];
		d_index[8192 + bid] = index[0];
	}
	//Need to remember the correct index
}


__global__ void atomsLight_kernel(float *d_fog, float *d_win_dark, int *d_index, float *d_im, float *d_atomsLight, float *d_atmos, int width, int height, int channel, int radius){
	int 	tid = threadIdx.x;
	//int 	bid = blockIdx.x;
	int   	it;
	float 	max;
	int 	max_index;

	__shared__ float temp[256];

	if (tid == 0)
	{
		max = d_win_dark[0];
		max_index = d_index[8192];//height*width/512
		for (it = 0; it<height*width / 1024 / 512; it++)
		{
			temp[it] = d_win_dark[it];
			temp[128 + it] = d_index[it + 8192];
		}
		__syncthreads();
		for (it = 0; it<height*width / 1024 / 512; it++)
		{
			if (max<temp[it])
			{
				max = temp[it];
				max_index = temp[128 + it];
			}
		}
		//如果d_atmos[0]为0，则说明是视频传入的第一帧或是图像去雾
		if (d_atmos[0] == 0)
		{
			for (int k = 0; k < channel; ++k)
			{
				d_atomsLight[k] = d_fog[max_index * channel + k];
			}
			/*d_atomsLight[0] = 194;
			d_atomsLight[1] = 205;
			d_atomsLight[2] = 213;*/
		}
		else
		{
			for (int k = 0; k < channel; ++k)
			{
				d_atomsLight[k] = d_fog[max_index * channel + k] * 0.8 + d_atmos[k] * 0.2;
			}
		}
		for (int k = 0; k < channel; ++k)
		{
			d_atmos[k] = d_atomsLight[k];
		}
	}

}

__global__ void atomsLight_kernel_divide(float *d_fog, float *d_im, float *d_atomsLight, int width, int height, int channel){
	int 	idx = threadIdx.x;
	int 	idy = blockIdx.x;
	int		id = idy*blockDim.x + idx;

	if (id<height*width){
		d_im[id*channel] = d_fog[id*channel + 0] / d_atomsLight[0];
		d_im[id*channel + 1] = d_fog[id*channel + 1] / d_atomsLight[1];
		d_im[id*channel + 2] = d_fog[id*channel + 2] / d_atomsLight[2];
	}
}


__global__ void t_initial_kernel(float *d_dark_mat, float *d_tDown, int width, int height, int channel){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float w = 0.95;
	if (id<height*width){
		d_tDown[id] = ((1.0f - w*d_dark_mat[id]) > 0.1) ? (1.0f - w*d_dark_mat[id]) : 0.1;
	}
}

__global__ void clear_kernel(float *d_fog, unsigned char *d_dehaze, float *d_atomsLight, float *d_filtered, int width, int height, int channel, int radius, int atmos_correct)
{

	int		k;
	float 	cha, inten, alpha2, tDown, clear;
	float 	d_atomsLight_ave;
	int		id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id<height*width)
	{
		//cha = 0;
		//for (k = 0; k<channel; k++)
		//{
		//	cha += d_fog[id*channel + k];
		//	d_atomsLight_ave += d_atomsLight[k];
		//}
		//inten = cha / 3;
		//cha = fabs(inten - d_atomsLight_ave);
		//alpha2 = mymin(mymax(atmos_correct / cha, 1.0f)*mymax(d_filtered[id], 0.1), 1);  //当atmos_correct为0时，不进行大气光校正

		//tDown = mymax(0.1f, alpha2); 
		tDown = mymax(0.1f, d_filtered[id]); 
		
		for(k = 0;k < channel; k++)
		{			
			clear = (d_fog[id * channel + k] - d_atomsLight[k]) / tDown + d_atomsLight[k];
			d_dehaze[id * channel + k] = round_me(clear);	
		}
	}
}