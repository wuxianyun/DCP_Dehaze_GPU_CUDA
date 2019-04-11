//boxfilter

#define TILE_DIM 16
#define BLOCKSIZE 128

__global__ void d_boxfilter_x_global(float *src, float *dst, int width, int height, int r)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = 1;
	int num = (width + 2 * r + 2 * BLOCKSIZE - 1) / (2 * BLOCKSIZE);		//每一个线程块被BLOCKSIZE*2分割成了num个segment
	int len = num * 2 * BLOCKSIZE;
	int extra = len - r - width;
	float scale = 1.0f / (float)((r << 1) + 1);

	__shared__ float sum[35]; sum[0] = 0;

	extern __shared__ float temp[];

	if (bid < height)
	{
	
		for (int i = tid; i < r; i += BLOCKSIZE)
		{
			temp[i] = src[bid*width + 0];								
		}
		

		for (int i = tid; i < width; i += BLOCKSIZE)
		{
			temp[r + i] = src[bid * width + i];
		}
		

		for (int i = tid; i < extra; i += BLOCKSIZE)
		{
			temp[r + width + i] = src[(bid + 1) * width - 1];			
		}
		__syncthreads();


		for (int cnt = 0; cnt < num; ++cnt)							
		{
			int bias = cnt * BLOCKSIZE * 2;

			for (int j = BLOCKSIZE; j > 0; j >>= 1)
			{
				if (tid < j)
				{
					int ai = bias + offset * (2 * tid + 1) - 1;
					int bi = bias + offset * (2 * tid + 2) - 1;
					temp[bi] += temp[ai];
				}
				offset *= 2;
				__syncthreads();
			}
			
			if (tid == 0)
			{
				sum[cnt + 1] = temp[(cnt + 1) * BLOCKSIZE * 2 - 1] + sum[cnt]; 
				temp[(cnt + 1) * BLOCKSIZE * 2 - 1] = 0;
			}
			__syncthreads();
			for (int j = 1; j < (BLOCKSIZE * 2); j *= 2)
			{
				offset >>= 1;
				if (tid < j)
				{
					int ai = bias + offset * (2 * tid + 1) - 1;
					int bi = bias + offset * (2 * tid + 2) - 1;

					float t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
				__syncthreads();
			}
		}
		for (int i = tid; i < width; i += BLOCKSIZE)
		{
			float sum_box = temp[i + 2 * r + 1] + sum[(i + 2 * r + 1) / (BLOCKSIZE * 2)] - temp[i] - sum[i / (BLOCKSIZE * 2)];		//sum只是第i + 2 * r + 1之前的所有元素之和不包括第i + 2 * r + 1个元素
			dst[bid * width + i] = sum_box * scale;

		}
	}
}

