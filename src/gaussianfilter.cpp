#include "gaussianfilter.h"
#include "utilities.h"
#include <glog/logging.h>
#include <cmath>
using namespace std;

void GaussianFilter::Run(
	unsigned char *image_in, unsigned char* image_out,
	float sigma, int radius, int w, int h, int bpp)
{
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";

	auto kernel = CreateGaussianKernel(sigma, radius+1);
	int buf_size = radius*2+1;
	unique_ptr<float[]> mid(new float[buf_size]());
	const int line_stride = bpp*w;
	for (int y = radius; y < h-radius; ++y) {
		for (int x = bpp*radius; x < line_stride-bpp*radius; ++x) {
			for (int k = -radius; k <= radius; ++k) {
				for (int i = -radius; i <= radius; ++i) {
					mid[k+radius] += kernel[i+radius] * image_in[(y+i)*line_stride+(x+bpp*k)];
				}
			}
			int pixel_output = (int)inner_product(mid.get(), mid.get()+buf_size, kernel.get(), 0.0f);
			image_out[y*line_stride+x] = ClampToUint8(pixel_output);
			fill(mid.get(), mid.get()+buf_size, 0.0f);
		}
	}
}
