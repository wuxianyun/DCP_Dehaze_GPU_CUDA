#include<iostream>
#include<opencv2/opencv.hpp>
#include"dehaze_resource_init.h"
#include"dehaze_resource_release.h"

extern "C"
void DehazeGPU(const cv::Mat srcimg, const int radius, const int atmos_correct, float * atmos, struct dehaze_resource *img_resource, cv::Mat dstimg);
