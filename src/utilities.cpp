#include "utilities.h"

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
