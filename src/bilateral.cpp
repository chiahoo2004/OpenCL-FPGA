#include "bilateralfilter.h"
#include <cmath>
#include <memory>
#include <algorithm>
#include <IL/il.h>
#include <glog/logging.h>

using namespace std;

#define DEBUG 0

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
	vector<vector<double> > kernel;
	kernel.resize(size);
	for(int i=0; i<size; ++i) 
		kernel[i].resize(size);


	// Bilateral filter
	BilateralFilter bf;
	bf.Run(color_img_ptr, color_img_buffer.get(), kernel, w, h, bpp);

	copy(color_img_buffer.get(), color_img_buffer.get()+w*h*bpp, color_img_ptr);

	// store image
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[2]);
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	return 0;
}

