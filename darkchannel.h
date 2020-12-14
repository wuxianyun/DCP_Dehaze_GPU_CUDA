#ifndef DARKCHANNEL_H
#define DARKCHANNEL_H

extern "C"
void minfilter(float *d_fog, float *d_min_img, float *d_win_dark, float *d_temp, int width, int height, int channel, int radius, int BLOCKSIZE);

#endif