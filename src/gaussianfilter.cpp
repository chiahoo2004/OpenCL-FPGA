#include "gaussianfilter.h"
#include "utilities.h"
#include <glog/logging.h>
#include <cmath>
#include <memory>
using namespace std;

template<class Float = float> unique_ptr<Float[]> CreateGaussianKernel(Float sigma, int radius)
{
	int length = radius*2+1;
	unique_ptr<Float[]> table(new Float[length]);
	const Float denominator_inverse = -1.0f / (2.0f * sigma * sigma);
	Float sum = 0;
	for (int i = 0; i < length; ++i) {
		int diff = i-radius;
		table[i] = exp(diff*diff*denominator_inverse);
		sum += table[i];
	}
	sum = 1 / sum;
	for (int i = 0; i < length; ++i) {
		table[i] *= sum;
	}
	return table;
}

void GaussianFilter::SetDimension(const int w, const int h, const int channel)
{
	Filter::SetDimension(w, h, channel);
}

void GaussianFilter::Run(const float *image_in, float* image_out)
{
	const int w = w_;
	const int h = h_;
	const int radius = param_.radius;
	const int bpp = channel_;
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";

	auto kernel = CreateGaussianKernel(param_.spacial_sigma, radius);
	int buf_size = radius*2+1;
	vector<vector<vector<float>>> mid;
	mid.resize(h);
	for(int i=0; i<h; ++i) {
		mid[i].resize(w);
		for (int j = 0; j < w; ++j) {
			mid[i][j].resize(bpp);
		}
	}
	const int line_stride = bpp*w;
	for (int y = radius; y < h-radius; ++y) {
		for (int a = 0; a < bpp; ++a){
			for (int b = 0; b < 2*radius+1; ++b){
				for (int c = -radius; c < radius; ++c){
					mid[y][b][a] += kernel[c+radius] * image_in[(y+c)*line_stride+(a+bpp*b)];;
				}
			}
		}
		for (int x = bpp*(radius+1); x < line_stride-bpp*radius; ++x) {
			for (int c = -radius; c <= radius; ++c) {
				mid[y][ (x/bpp)+radius ][ x%bpp ] += kernel[c+radius] * image_in[(y+c)*line_stride+(x%bpp+bpp*(x/bpp+radius))];
			}

			float pixel_output = 0;
			for (int c = -radius; c < radius; ++c)
			{
				pixel_output += kernel[c+radius] * mid[y][ (x/bpp)+c ][ x%bpp ];
			}
			image_out[y*line_stride+x] = pixel_output;
		}
	}
}
