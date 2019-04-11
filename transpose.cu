__global__ void transpose(float *odata, float *idata, 
int width, int height)
{
__shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];	
int xIndex = blockIdx.x * BLOCK _DIM + threadIdx.x;
int yIndex = blockIdx.y * BLOCK _DIM + threadIdx.y;
if (xIndex<width && yIndex<height)
{
	int index_in = xIndex + (yIndex)*width;
	tile[threadIdx.y][threadIdx.x] = idata[index_in];
}
__syncthreads();
int xIndex_new = blockIdx.y * BLOCK _DIM + threadIdx.x;
int yIndex_new = blockIdx.x * BLOCK _DIM + threadIdx.y;
if (xIndex_new<height && yIndex_new<width)
{
	int index_out = xIndex_new + (yIndex_new)*height;
	odata[index_out] = tile[threadIdx.x][threadIdx.y];
}
}
