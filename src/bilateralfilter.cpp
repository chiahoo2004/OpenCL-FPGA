#include "bilateralfilter.h"
#include "utilities.h"
#include <iostream>
#include <IL/il.h>
#include <glog/logging.h>
#include <cmath>
#include <memory>
using namespace std;

static void createKernel(const float *image_in, vector<vector<float>>& kernel, float sigma_s, float sigma_r, int a, int b, int w, int h, int bpp)
{
	const int line_stride = bpp*w;
	// set standard deviation to 1.0
	float s = 2.0 * sigma_s * sigma_s;
	float r = 2.0 * sigma_r * sigma_r;

	// sum is for normalization
	float sum = 0.0;

	int size = kernel.size();
	int offset = (size-1)/2;

	// generate nxn kernel
	for (int x = -offset; x <= offset; x++) {
		for(int y = -offset; y <= offset; y++) {
			float spatial = exp(-(x*x+y*y)/s);
			float range_diff = image_in[a*bpp+b*line_stride]-image_in[(a+x)*bpp+(b+y)*line_stride];
			float range = exp(-range_diff * range_diff / r);
			kernel[x + offset][y + offset] = range * spatial;
			sum += kernel[x + offset][y + offset];
		}
	}

	for(int i = 0; i < size; ++i)
		for(int j = 0; j < size; ++j)
			kernel[i][j] /= sum;
}

void BilateralFilter::SetDimension(const int w, const int h, const int channel)
{
	Filter::SetDimension(w, h, channel);
}

void BilateralFilter::Run(const float *image_in, float *image_out)
{
	const int bpp = channel_;
	const int w = w_;
	const int h = h_;
	const int radius = param_.radius;
	const float spacial_sigma = param_.spacial_sigma;
	const float color_sigma = param_.color_sigma;
	const int size = radius*2+1;
	const int offset = radius;
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";

	vector<vector<float>> kernel(size);
	for (auto& row: kernel) {
		row.resize(size);
	}

	vector<float> mid(size,0);
	const int line_stride = bpp*w;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = bpp*offset; x < line_stride-bpp*offset; ++x) {
			float image = 0.0f;
			createKernel(image_in, kernel, spacial_sigma, color_sigma, x, y, w, h, bpp);
			for (int a = -offset; a <= offset; a++) {
				for(int b = -offset; b <= offset; b++) {
					image +=  kernel[a+offset][b+offset]   *  image_in[(y+a)*line_stride+(x+b*bpp)];
				}
			}
			image_out[y*line_stride+x] = image;
		}
	}
}
