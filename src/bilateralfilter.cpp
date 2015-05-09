#include "bilateralfilter.h"
#include "utilities.h"
#include <glog/logging.h>
using namespace std;

void BilateralFilter::createKernel(unsigned char *image_in, vector<vector<double> >& kernel, int a, int b, int w, int h, int bpp)
{
	const int line_stride = bpp*w;
	// set standard deviation to 1.0
	double sigma_s = 10.0;
	double sigma_r = 20.0;
	double s = 2.0 * sigma_s * sigma_s;
	double r = 2.0 * sigma_r * sigma_r;

	// sum is for normalization
	double sum = 0.0;

	int size = kernel.size();
	int offset = (size-1)/2;

	// generate nxn kernel
	for (int x = -offset; x <= offset; x++) {
		for(int y = -offset; y <= offset; y++) {
			double spatial = exp(-(x*x+y*y)/s);
			int range_diff = image_in[a*bpp+b*line_stride]-image_in[(a+x)*bpp+(b+y)*line_stride];
			double range = exp(-range_diff * range_diff / r);
			kernel[x + offset][y + offset] = range * spatial;
			sum += kernel[x + offset][y + offset];
		}
	}

	for(int i = 0; i < size; ++i)
		for(int j = 0; j < size; ++j)
			kernel[i][j] /= sum;

#ifndef NDEBUG
	for(int i = 0; i < size; ++i) {
		for(int j = 0; j < size; ++j) {
			DLOG(INFO)<<kernel[i][j]<<"     ";
		}
		LOG(INFO)<<endl;
	}
#endif
}

void BilateralFilter::Run(unsigned char *image_in, unsigned char* image_out, vector<vector<double> >& kernel, int w, int h, int bpp)
{
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";

	int size = kernel.size();
	int offset = (size-1)/2;

	vector<double> mid(size,0);
	const int line_stride = bpp*w;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = bpp*offset; x < line_stride-bpp*offset; ++x) {
			int image=0;
            DLOG(INFO)<<"image_in["<<y*line_stride+x<<"] = "<< (double) image_in[y*line_stride+x] <<endl;
            createKernel(image_in, kernel, x, y, w, h, bpp);
#ifndef NDEBUG
            for (int a = -offset; a <= offset; a++) {
                for(int b = -offset; b <= offset; b++) {
                    image +=  kernel[a+offset][b+offset]   *  image_in[(y+a)*line_stride+(x+b*bpp)];
                    LOG(INFO)<<"kernel["<<a+offset<<"]["<<b+offset<<"]   *   image_in["<<(y+a)*line_stride+(x+b*bpp)<<"]"<<endl;
                }
            }
#endif
            image_out[y*line_stride+x] = ClampToUint8(abs(image)*1);
#ifndef NDEBUG
            DLOG(INFO)<<"image_out["<<y*line_stride+x<<"] = "<< ClampToUint8(abs(image)*1)<<endl;
#endif

		}
	}
}
