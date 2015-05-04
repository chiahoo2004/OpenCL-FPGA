#include <iostream>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <IL/il.h>
#include <glog/logging.h>

using namespace std;

template <class SignedIntType=int> SignedIntType ClampToUint8(SignedIntType x)
{
	const SignedIntType mask = 0xff;
	return (x&~mask)? ((~x)>>(sizeof(SignedIntType)*8-1) & mask): x;
}

void createKernel(vector<double>& kernel)
{
    // set standard deviation to 1.0
    double sigma = 1.0;
    double x, s = 2.0 * sigma * sigma;
 
    // sum is for normalization
    double sum = 0.0;

    int size = kernel.size();
    int offset = (size-1)/2;
 
/*    // generate nxn kernel
    for (int x = -2; x <= 2; x++)
    {
	for(int y = -2; y <= 2; y++)
	{
		r = sqrt(x*x + y*y);
		kernel[x + 2][y + 2] = (exp(-(r*r)/s))/(M_PI * s);
		sum += gKernel[x + 2][y + 2];
	}
    }
*/ 
    	for(int x = -offset; x <= offset; ++x)
    	{
    		kernel[x+offset] = (exp(-(x*x)/s))/sqrt(M_PI * s);
    		sum += kernel[x];
    	}

	// normalize the Kernel
	for(int x = 0; x < 5; ++x)
		kernel[x] /= sum;
 
}

struct GaussianFilter {
	void FillBoundary(unsigned char* image_out, int offset, int w, int h, int bpp)
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

	void Run(unsigned char *image_in, unsigned char* image_out,vector<double>& kernel, int w, int h, int bpp)
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
};


#define IL_CHECK_ERROR() {auto e = ilGetError();CHECK_EQ(e, IL_NO_ERROR)<<e;}
int main(int argc, char const* argv[])
{
	google::InitGoogleLogging(argv[0]);
	FLAGS_logtostderr = true;
	ilInit();

	LOG(INFO) << "Using devil library version " << ilGetInteger(IL_VERSION_NUM);

	// Allocate images
	ILuint image;
	ilGenImages(1, &image);
	ilBindImage(image);
	IL_CHECK_ERROR();

	// Read image
	ilLoadImage("lena_color.bmp");
	IL_CHECK_ERROR();
	auto bpp = ilGetInteger(IL_IMAGE_BPP); // bpp = byte per pixels
	auto w = ilGetInteger(IL_IMAGE_WIDTH);
	auto h = ilGetInteger(IL_IMAGE_HEIGHT);
	LOG(INFO) << "Load image width = " << w << ", height = " << h;
	ILubyte *color_img_ptr = ilGetData();
	vector<ILubyte> color_img_edge(w*h*bpp);
	IL_CHECK_ERROR();

	int size = 5;
	vector<double> kernel;
	kernel.resize(size);

	// kernel
	createKernel(kernel);


	// Gaussian filter
	GaussianFilter gf;
	gf.Run(color_img_ptr, color_img_edge.data(), kernel, w, h, bpp);

	


	copy(color_img_edge.begin(), color_img_edge.end(), color_img_ptr);

	// store image
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage("lena_gaussian.bmp");
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	return 0;
}
