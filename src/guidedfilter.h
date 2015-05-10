#pragma once
#include <vector>
using std::vector;

struct GuidedFilter {
	void FillBoundary(unsigned char* image_out, int offset, int w, int h, int bpp);
	void Run(
		unsigned char *image_in, unsigned char* image_out, unsigned char* I,
		vector<vector<vector<double> > >& a, vector<vector<vector<double> > >& b, 
		int windowSize, int w, int h, int bpp);
};
