#include "gaussianfilter.h"
#include <algorithm>
#include <glog/logging.h>
using namespace std;

template <class SignedIntType=int> SignedIntType ClampToUint8(SignedIntType x)
{
	const SignedIntType mask = 0xff;
	return (x&~mask)? ((~x)>>(sizeof(SignedIntType)*8-1) & mask): x;
}

void GaussianFilter::FillBoundary(unsigned char* image_out, int offset, int w, int h, int bpp)
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

void GaussianFilter::Run(unsigned char *image_in, unsigned char* image_out,vector<double>& kernel, int w, int h, int bpp)
{
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";


//		int kernel[l] = {-1,2,-1};
	int size = kernel.size();
	int offset = (size-1)/2;
	FillBoundary(image_out, offset, w, h, bpp);

/*
	for (int i = 0; i < size; ++i)
	{
		kernel[0]=-1;
		kernel[1]=2;
		kernel[2]=-1;
	}
*/
/*
	for (int i = 0; i < size; ++i)
		LOG(INFO) <<kernel[i]<<endl;
*/
//		double* image_in = image_1;
//		double* image_out = image_2;

	
	vector<double> mid(size,0);
	const int line_stride = bpp*w;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = bpp*offset; x < line_stride-bpp*offset; ++x) {
			
			for (int k = 0; k < size; ++k) {
				for (int i = 0; i < size; ++i) {
					int j = -offset; 
					mid[k] += kernel[i] * image_in[(y-offset+i)*line_stride+(x+bpp*(k-offset))];
//						LOG(INFO) << "mid["<<k<<"] += "<<"kernel["<<i<<"] ("<<kernel[i]<<") * image_in["<<(y-offset+i)*line_stride+(x+bpp*(k-offset))<<"] ("
//							<<(double)image_in[(y-offset+i)*line_stride+(x+bpp*(k-offset))]<<" )"<<endl;
				}

//					if( mid[k]!=0 )
//						LOG(INFO) << "mid["<<k<<"] = "<<mid[k];

			}
			

			int image=0;
			for (int k = 0; k < size; ++k) {
				image += mid[k]*kernel[k];
			}

			image_out[y*line_stride+x] = ClampToUint8(abs(image)*1);
			
			fill(mid.begin(), mid.end(), 0);

		}
	}
}
