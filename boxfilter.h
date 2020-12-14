#ifndef BOXFILTER_H
#define BOXFILTER_H

extern "C"
void boxfilter(float *d_src, float *d_dest, float *d_temp, float *d_temp1, int height, int width, int radius);

#endif