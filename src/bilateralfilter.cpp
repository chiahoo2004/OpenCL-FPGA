#include "bilateralfilter.h"
#include <algorithm>
#include <glog/logging.h>
using namespace std;

#define DEBUG 0

template <class SignedIntType=int> SignedIntType ClampToUint8(SignedIntType x)
{
	const SignedIntType mask = 0xff;
	return (x&~mask)? ((~x)>>(sizeof(SignedIntType)*8-1) & mask): x;
}

void BilateralFilter::FillBoundary(unsigned char* image_out, int offset, int w, int h, int bpp)
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

//	LOG(INFO)<<"a:"<<a<<endl;
//	LOG(INFO)<<"b:"<<b<<endl;

	// generate nxn kernel
	for (int x = -offset; x <= offset; x++)
	{
		for(int y = -offset; y <= offset; y++)
		{
			double spatial = exp(-(x*x+y*y)/s);
            			double range = exp(   -   (   image_in[   a*bpp+b*line_stride   ]-image_in[     (a+x)*bpp+(b+y)*line_stride     ]    ) *
                       			 (   image_in[   a*bpp+b*line_stride   ]-image_in[     (a+x)*bpp+(b+y)*line_stride     ]   )   /   r   );
			
//    		LOG(INFO)<<x + offset<<endl;
//			LOG(INFO)<<y + offset<<endl;

			kernel[x + offset][y + offset] = range * spatial;

//			cout<<"pause"<<endl;
//			fgetc(stdin);


			sum += kernel[x + offset][y + offset];
		}
	}

	for(int i = 0; i < size; ++i)
		for(int j = 0; j < size; ++j)
			kernel[i][j] /= sum;

	if(DEBUG)
	{
		for(int i = 0; i < size; ++i)
		{
			for(int j = 0; j < size; ++j)
			{
				LOG(INFO)<<kernel[i][j]<<"     ";
			}
			LOG(INFO)<<endl;
		}
	}
}

void BilateralFilter::Run(unsigned char *image_in, unsigned char* image_out, vector<vector<double> >& kernel, int w, int h, int bpp)
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
			
			int image=0;

            if(DEBUG)
            LOG(INFO)<<"image_in["<<y*line_stride+x<<"] = "<< (double) image_in[y*line_stride+x] <<endl;

            createKernel(image_in, kernel, x, y, w, h, bpp);

//				cout<<"pause"<<endl;
//				fgetc(stdin);

            for (int a = -offset; a <= offset; a++)
            {
                for(int b = -offset; b <= offset; b++)
                {	
                    image +=  kernel[a+offset][b+offset]   *  image_in[(y+a)*line_stride+(x+b*bpp)];
                    if(DEBUG)
                    LOG(INFO)<<"kernel["<<a+offset<<"]["<<b+offset<<"]   *   image_in["<<(y+a)*line_stride+(x+b*bpp)<<"]"<<endl;
                }
            }


            image_out[y*line_stride+x] = ClampToUint8(abs(image)*1);
            if(DEBUG)
            LOG(INFO)<<"image_out["<<y*line_stride+x<<"] = "<< ClampToUint8(abs(image)*1)<<endl;

		}
	}
}
