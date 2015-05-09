#pragma once
#include <cmath>
#include <algorithm>
using namespace std;

template <class Int=int> Int ClampToUint8(Int x)
{
	const Int mask = 0xff;
	return (x&~mask)? ((~x)>>(sizeof(Int)*8-1) & mask): x;
}

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
