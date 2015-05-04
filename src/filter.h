#pragma once
#include <vector>
using std::vector;

struct GaussianFilter {
	void FillBoundary(unsigned char* image_out, int offset, int w, int h, int bpp);
	void Run(unsigned char *image_in, unsigned char* image_out, vector<double>& kernel, int w, int h, int bpp);
};

