#ifndef DEHAZE_KERNEL_H
#define DEHAZE_KERNEL_H

__device__ float mymin(float a, float b);
__device__ float mymax(float a, float b);
__device__ unsigned char round_me(float x);

__global__ void matrix_add(float *src1, float *src2, float *dst, int row, int column);
__global__ void matrix_subtract(float *src1, float *src2, float *dst, int row, int column);
__global__ void matrix_add_all(float *src, float *dst, int row, int column, float sum);
__global__ void matrix_divide(float *src, float *dst, int row, int column, float div);
__global__ void matrix_dot_divide(float *src1, float *src2, float *dst, int row, int column);
__global__ void matrix_dot_multiple(float *src1, float *src2, float *dst, int row, int column);

__global__ void float_fog_kernel(unsigned char *src, float *dst, float *d_foggy_gray, int width, int height, int channel);
__global__ void transpose(float *odata, float *idata, int width, int height);
__global__ void MaxReductionkernel(float *d_win_dark, float *d_im_dark, int width, int height, int *d_index);
__global__ void MaxReductionkernelTwo(float *d_im_dark, float *d_win_dark, float *d_foggy_gray, int width, int height, int *d_index);
__global__ void atomsLight_kernel(float *d_fog, float *d_win_dark, int *d_index, float *d_im, float *d_atomsLight, float *d_atmos, int width, int height, int channel, int radius);
__global__ void atomsLight_kernel_divide(float *d_fog, float *d_im, float *d_atomsLight, int width, int height, int channel);
__global__ void t_initial_kernel(float *d_dark_mat, float *d_tDown, int width, int height, int channel);

__global__ void matrix_cal_a(float *mean_I, float *mean_p, float *mean_II, float *mean_Ip, float *a, float eps, int row, int column);
__global__ void matrix_cal_b(float *mean_I, float *mean_p, float *mean_II, float *mean_Ip, float *b, float eps, int row, int column);
__global__ void matrix_cal_q(float *mean_a, float *mean_b, float *I, float *q, int width, int height);

__global__ void clear_kernel(float *d_fog, unsigned char *d_dehaze, float *d_atomsLight, float *d_filtered, int width, int height, int channel, int radius, int kk);
#endif