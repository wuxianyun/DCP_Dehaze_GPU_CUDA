#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "darkchannel.h"
#include "dehaze_kernel.h"
#include "boxfilter.h"
#include <sys/time.h>//clock

#define BLOCKSIZE 256

using namespace std;




static void HandleError(cudaError_t err, const char *file, int line){
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

void time_printf(double mseca)
{
	double msecb;
	struct timeval tb;
	gettimeofday( &tb, NULL);
	msecb = tb.tv_sec*1000.0 + tb.tv_usec/1000.0;
	msecb -= mseca;
	printf("  %.10f\n", msecb);   
}

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

int main(int argc, char *argv[])
{

	cudaSetDevice(7);
	int height, width, channel, radius;
	char img_name[20], dehaze_img_name[20];
	float atmos_correct = 0.0;
	float eps = 1e-05f;   //导向滤波修正参数

	FILE *fp_in, *fp_out;
	//ofstream out("out.txt");
	unsigned char   *ori, *d_ori;
	unsigned char	*dehaze, *d_dehaze;
	
	float			*d_fog;
	float           *d_min_img;
	float			*d_win_dark;
	int				*d_index;
	float			*d_im_dark;
	float           *d_dark_mat;
	float			*d_im;
	float			*d_tDown;
	float			*d_foggy_gray;
	float			*d_atmosLight;
	float           *d_atmos;

	float			*d_mean_I;
	float			*d_mean_p;
	float           *d_mean_Ip;
	float           *d_cov_Ip;
	float           *d_mean_II;
	float           *d_var_I;
	float           *d_a;
	float           *d_b;
	float           *d_mean_a;
	float           *d_mean_b;
	float			*d_box_temp;
	float			*d_filtered;
	float			*d_t;
	float			*d_temp;
	float           *d_temp1;
	
	float           *atmos;
	float           *hhh;

	/*if (argc != 7)
	{
		printf("6 parameters are needed:\n");
		printf(" Input image name, output image name, height, width, channel, radius.");
		exit(0);
	}
	else*/
	{
		strcpy(img_name, argv[1]);//Input image name
		strcpy(dehaze_img_name, argv[2]);//Output dehaze image name
		height = atoi(argv[3]); //Row of the input image 
		width = atoi(argv[4]);//Col of the input image
		channel = atoi(argv[5]);//RGB or gray
		radius = atoi(argv[6]); //滤波半径
	}

	cudaHostAlloc((void **)&ori, sizeof(unsigned char)*width*height*channel, cudaHostAllocMapped);
	cudaHostAlloc((void **)&dehaze, sizeof(unsigned char)*width*height*channel, cudaHostAllocMapped);
	//cudaHostAlloc((void **)&dehaze, sizeof(unsigned char)*width*height, cudaHostAllocMapped);
	cudaHostAlloc((void **)&hhh, sizeof(float)*width*height, cudaHostAllocMapped);
	cudaHostAlloc((void **)&atmos, sizeof(float)*3, cudaHostAllocMapped);

	HANDLE_ERROR(cudaMalloc((void **)&d_ori, height*width*channel*sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void **)&d_fog, sizeof(float)*width*height*channel));
	HANDLE_ERROR(cudaMalloc((void **)&d_dehaze, sizeof(unsigned char)*width*height*channel));
	HANDLE_ERROR(cudaMalloc((void **)&d_min_img, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_win_dark, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_im_dark, sizeof(float)*width*height));

	HANDLE_ERROR(cudaMalloc((void **)&d_dark_mat, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_index, sizeof(int)*width*height));//2048 is enough
	HANDLE_ERROR(cudaMalloc((void **)&d_im, sizeof(float)*width*height*channel));
	HANDLE_ERROR(cudaMalloc((void **)&d_temp, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_atmosLight, sizeof(float) * 3));
	HANDLE_ERROR(cudaMalloc((void **)&d_atmos, sizeof(float) * 3));
	HANDLE_ERROR(cudaMalloc((void **)&d_filtered, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_t, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_foggy_gray, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_tDown, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_mean_I, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_mean_p, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_mean_Ip, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_cov_Ip, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_mean_II, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_var_I, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_a, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_b, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_mean_a, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_mean_b, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_box_temp, sizeof(float)*width*height));
	HANDLE_ERROR(cudaMalloc((void **)&d_temp1, sizeof(float)*width*height));
	
	

	//读图并从host端读到device端
	fp_in = fopen(img_name, "rb");
	fread(ori, width*height*channel, 1, fp_in);
	fclose(fp_in);
	
	struct timeval ta;
	double mseca;  
	gettimeofday( &ta, NULL);
	mseca = ta.tv_sec*1000.0 + ta.tv_usec/1000.0;
	
	HANDLE_ERROR(cudaMemcpy(d_ori, ori, sizeof(unsigned char)*width*height*channel, cudaMemcpyHostToDevice));

	//GPU
	{
		//将uchar型的输入图像数据转换为float型，方便后面进行计算
		float_fog_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_ori, d_fog, d_foggy_gray, width, height, channel);
		
		//求解暗通道图
		minfilter(d_fog, d_min_img, d_win_dark, d_temp, width, height, channel, radius, BLOCKSIZE);
		MaxReductionkernel << <(height * width ) / 1024 +1, 512 >> >(d_win_dark, d_im_dark, width, height, d_index);//Find 2048 max value into d_index
		MaxReductionkernelTwo << < height * width / 1024 / 512 + 1, 256 >> >(d_im_dark, d_win_dark, d_foggy_gray, width, height, d_index);
		atomsLight_kernel << <1, 1 >> >(d_fog, d_win_dark, d_index, d_im, d_atmosLight, d_atmos, width, height, channel, radius);
		atomsLight_kernel_divide << <(height*width + BLOCKSIZE-1) / BLOCKSIZE, BLOCKSIZE >> >(d_fog, d_im, d_atmosLight, width, height, channel);

		//计算初始透射率图
		minfilter(d_im, d_min_img, d_dark_mat, d_temp, width, height, channel, radius, BLOCKSIZE);
		t_initial_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_dark_mat, d_tDown, width, height, channel);

		//Guided Filter
		
		boxfilter(d_foggy_gray, d_mean_I, d_temp, d_temp1, height, width, radius * 5);
		boxfilter(d_tDown, d_mean_p, d_temp, d_temp1, height, width, radius * 5);
		matrix_dot_multiple << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_foggy_gray, d_tDown, d_box_temp, height, width);

		boxfilter(d_box_temp, d_mean_Ip, d_temp, d_temp1, height, width, radius * 5);
		matrix_dot_multiple << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_foggy_gray, d_foggy_gray, d_box_temp, height, width);

		boxfilter(d_box_temp, d_mean_II, d_temp, d_temp1, height, width, radius * 5);
		matrix_cal_a << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_mean_I, d_mean_p, d_mean_II, d_mean_Ip, d_a, eps, height, width);

		matrix_cal_b << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_mean_I, d_mean_p, d_mean_II, d_mean_Ip, d_b, eps, height, width);

		boxfilter(d_a, d_mean_a, d_temp, d_temp1, height, width, radius * 5);
		
		boxfilter(d_b, d_mean_b, d_temp, d_temp1, height, width, radius * 5);
		matrix_cal_q << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_mean_a, d_mean_b, d_foggy_gray, d_filtered, width, height);

		//恢复原始景物光线
		clear_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_fog, d_dehaze, d_atmosLight, d_filtered, width, height, channel, radius, atmos_correct);
	}

	HANDLE_ERROR(cudaMemcpy(dehaze, d_dehaze, width*height*channel * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(atmos, d_atmosLight, 3 * sizeof(float), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(hhh, d_filtered, width*height * sizeof(float), cudaMemcpyDeviceToHost));
	
	
	time_printf(mseca);
	
	//for (int i = 0; i < height; i++)
	//{
	//	for (int j = 0; j < width; ++j)
	//	{
	//		dehaze[i*width + j] = hhh[i*width + j] * 255;
	//	}
	//}
	
	//for (int i = 0; i < 3; ++i)
	//{
	//	printf("%f\n", atmos[i]);
	//}

	/*if (out.is_open())
	{
		for (int i = 0; i < (height); ++i)
		{
			for (int j = 0; j < (width); ++j)
			{
				out << hhh[i*width + j] << '\t';
			}
			out << '\n';
		}
		out.close();
	}*/

	fp_out = fopen(dehaze_img_name, "wb");
	fwrite(dehaze, height*width*channel, 1, fp_out);
	fclose(fp_out);

	cudaFreeHost(ori);
	cudaFreeHost(dehaze);
	cudaFreeHost(atmos);
	cudaFree(d_ori);
	cudaFree(d_fog);
	cudaFree(d_dehaze);
	cudaFree(d_min_img);
	cudaFree(d_win_dark);
	cudaFree(d_im_dark);
	cudaFree(d_index);
	cudaFree(d_dark_mat);
	cudaFree(d_im);
	cudaFree(d_tDown);
	cudaFree(d_foggy_gray);
	cudaFree(d_atmos);
	cudaFree(d_atmosLight);
	cudaFree(d_filtered);
	cudaFree(d_t);
	cudaFree(d_mean_I);
	cudaFree(d_mean_p);
	cudaFree(d_mean_Ip);
	cudaFree(d_cov_Ip);
	cudaFree(d_mean_II);
	cudaFree(d_var_I);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_mean_a);
	cudaFree(d_mean_b);
	cudaFree(d_box_temp);
	cudaFree(d_temp);
	cudaFree(d_temp1);
	cudaFree(hhh);

	
	return 0;
}
