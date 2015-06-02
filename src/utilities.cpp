#include "utilities.h"
#include <algorithm>
#include <cmath>
using namespace std;

int CeilDiv(const int a, const int b)
{
	return (a-1)/b+1;
}

template<class Float> unique_ptr<Float[]> CreateGaussianKernel(Float sigma, int radius)
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
template unique_ptr<float[]> CreateGaussianKernel(float sigma, int radius);
template unique_ptr<double[]> CreateGaussianKernel(double sigma, int radius);

template<class Float> unique_ptr<Float[]> GenerateGaussianTable(const Float sigma, const int length)
{
	unique_ptr<Float[]> table(new Float[length]);
	const Float denominator_inverse = -1.0f / (2.0f * sigma * sigma);
	for (int i = 0; i < length; ++i) {
		table.get()[i] = exp(i*i*denominator_inverse);
	}
	return std::move(table);
}
template unique_ptr<float[]> GenerateGaussianTable(const float sigma, const int length);
template unique_ptr<double[]> GenerateGaussianTable(const double sigma, const int length);

void FillBoundary(unsigned char *image_out, int offset, int w, int h, int bpp)
{
	const int line_stride = bpp*w;
	for (int y = 0; y < offset; ++y) {
		fill(image_out, image_out+line_stride, 0);
		image_out += line_stride;
	}
	for (int y = offset; y < h-offset; ++y) {
		fill(image_out, image_out+bpp*offset, 0);
		fill(image_out+line_stride-bpp*offset, image_out+line_stride, 0);
		image_out += line_stride;
	}
	for (int y = h-offset; y < h; ++y) {
		fill(image_out, image_out+line_stride, 0);
		image_out += line_stride;
	}
}
