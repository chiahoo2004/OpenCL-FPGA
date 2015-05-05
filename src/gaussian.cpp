#include "gaussianfilter.h"
#include <cmath>
#include <memory>
#include <algorithm>
#include <IL/il.h>
#include <glog/logging.h>

using namespace std;

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
    		sum += kernel[x+offset];
    	}

	// normalize the Kernel
	for(int x = 0; x < size; ++x)
		kernel[x] /= sum;
 
}

#define IL_CHECK_ERROR() {auto e = ilGetError();CHECK_EQ(e, IL_NO_ERROR)<<e;}
int main(int argc, char const* argv[])
{
	google::InitGoogleLogging(argv[0]);
	FLAGS_logtostderr = true;
	CHECK_EQ(argc, 3) << "Usage: <executable> <input> <output>";

	ilInit();

	LOG(INFO) << "Using devil library version " << ilGetInteger(IL_VERSION_NUM);

	// Allocate images
	ILuint image;
	ilGenImages(1, &image);
	ilBindImage(image);
	IL_CHECK_ERROR();

	// Read image
	ilLoadImage(argv[1]);
	IL_CHECK_ERROR();
	auto bpp = ilGetInteger(IL_IMAGE_BPP); // bpp = byte per pixels
	auto w = ilGetInteger(IL_IMAGE_WIDTH);
	auto h = ilGetInteger(IL_IMAGE_HEIGHT);
	LOG(INFO) << "Load image width = " << w << ", height = " << h;
	ILubyte *color_img_ptr = ilGetData();
	unique_ptr<ILubyte[]> color_img_buffer(new ILubyte[w*h*bpp]);
	IL_CHECK_ERROR();

	int size = 5;
	vector<double> kernel(size);

	// kernel
	createKernel(kernel);

	// Gaussian filter
	GaussianFilter gf;
	gf.Run(color_img_ptr, color_img_buffer.get(), kernel, w, h, bpp);

	copy(color_img_buffer.get(), color_img_buffer.get()+w*h*bpp, color_img_ptr);

	// store image
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[2]);
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	return 0;
}
