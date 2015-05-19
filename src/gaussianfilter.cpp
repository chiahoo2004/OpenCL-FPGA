#include "gaussianfilter.h"
#include "utilities.h"
#include <IL/il.h>
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

void GaussianFilter::Edge(unsigned char *image_in, unsigned char* image_out, float sigma, int radius, int w, int h, int bpp)
{
	unique_ptr<ILubyte[]> color_img_g1(new ILubyte[w*h*bpp]);
	unique_ptr<ILubyte[]> color_img_g2(new ILubyte[w*h*bpp]);
 	Run(image_in, color_img_g1.get(), sigma, radius, w, h, bpp);
 	Run(color_img_g1.get(), color_img_g2.get(), sigma, radius, w, h, bpp);
 	unique_ptr<float[]> color_img_diff1(new float[w*h*bpp]);
 	unique_ptr<float[]> color_img_diff2(new float[w*h*bpp]);
 	for (int i = 0; i < w*h*bpp; ++i)
 	{
 		color_img_diff1[i] = (float)image_in[i] - (float)color_img_g1[i];
 		color_img_diff2[i] = (float)color_img_g1[i] - (float)color_img_g2[i];
		float temp = 2*(color_img_diff1[i]) + 1*(color_img_diff2[i]) + image_in[i];
		image_out[i] = ClampToUint8((int)temp);
 	}
}

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
