#pragma once
#include <vector>
using std::vector;

struct BilateralFilter {
	void FillBoundary(unsigned char* image_out, int offset, int w, int h, int bpp);
	void createKernel(unsigned char *image_in, vector<vector<double> >& kernel, int a, int b, int w, int h, int bpp);
	void Run(unsigned char *image_in, unsigned char* image_out, vector<vector<double> >& kernel, int w, int h, int bpp);
};
