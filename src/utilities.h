#pragma once
#include <memory>
using std::unique_ptr;

template <class Int=int> Int ClampToUint8(Int x)
{
	const Int mask = 0xff;
	return (x&~mask)? ((~x)>>(sizeof(Int)*8-1) & mask): x;
}

int CeilDiv(const int a, const int b);

void FillBoundary(unsigned char *image_out, int offset, int w, int h, int bpp);

template<class Float> unique_ptr<Float[]> CreateGaussianKernel(Float sigma, int radius);
extern template unique_ptr<float[]> CreateGaussianKernel(float sigma, int radius);
extern template unique_ptr<double[]> CreateGaussianKernel(double sigma, int radius);

template<class Float> unique_ptr<Float[]> GenerateGaussianTable(const Float sigma, const int length);
extern template unique_ptr<float[]> GenerateGaussianTable(const float sigma, const int length);
extern template unique_ptr<double[]> GenerateGaussianTable(const double sigma, const int length);
