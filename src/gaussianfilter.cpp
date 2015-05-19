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
 		color_img_diff1[i] = image_in[i] - color_img_g1[i];
 		color_img_diff2[i] = color_img_g1[i] - color_img_g2[i];
		image_out[i] = 10*(color_img_diff1[i]) + 5*(color_img_diff2[i]); 
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

void GaussianFilter::RunImproved(
	unsigned char *image_in, unsigned char* image_out,
	float sigma, int radius, int w, int h, int bpp)
{
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";

	auto kernel = CreateGaussianKernel(sigma, radius+1);
	int buf_size = radius*2+1;
//	unique_ptr<float[]> mid(new float[buf_size]());
	vector<vector<vector<float> > > mid;
	mid.resize(h);
	for(int i=0; i<h; ++i) {
		mid[i].resize(w);
		for (int j = 0; j < w; ++j)
		{
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
//					mid[k+radius] += kernel[i+radius] * image_in[(y+i)*line_stride+(x+bpp*k)];
				mid[y][ (x/bpp)+radius ][ x%bpp ] += kernel[c+radius] * image_in[(y+c)*line_stride+(x%bpp+bpp*(x/bpp+radius))];
			}
			
//			int pixel_output = (int)inner_product(mid.get(), mid.get()+buf_size, kernel.get(), 0.0f);
			int pixel_output = 0;
			for (int c = -radius; c < radius; ++c)
			{
				pixel_output += kernel[c+radius] * mid[y][ (x/bpp)+c ][ x%bpp ];
			}
			image_out[y*line_stride+x] = ClampToUint8(pixel_output);
		}
	}
}
