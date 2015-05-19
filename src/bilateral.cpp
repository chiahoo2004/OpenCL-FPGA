#include "bilateralfilter.h"
#include <iostream>
#include <memory>
#include <IL/il.h>
#include <glog/logging.h>

using namespace std;

#define DEBUG 0

#define IL_CHECK_ERROR() {auto e = ilGetError();CHECK_EQ(e, IL_NO_ERROR)<<e;}
int main(int argc, char const* argv[])
{
	google::InitGoogleLogging(argv[0]);
	FLAGS_logtostderr = true;
    CHECK_EQ(argc, 4) << "Usage: <executable> <input> <output> <output>";
    
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

	int size;
	cout<<"<kernel size> : ";
	cin>>size;
	vector<vector<double> > kernel;
	kernel.resize(size);
	for(int i=0; i<size; ++i) 
		kernel[i].resize(size);
	double sigma_s;
	double sigma_r;
	cout<<"<sigma_s> <sigma_r> : ";
	cin>>sigma_s>>sigma_r;


	// Bilateral filter
	BilateralFilter bf;
	bf.Run(color_img_ptr, color_img_buffer.get(), kernel, sigma_s, sigma_r, w, h, bpp);

	copy(color_img_buffer.get(), color_img_buffer.get()+w*h*bpp, color_img_ptr);

	// store image
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[2]);
	IL_CHECK_ERROR();

	bf.Edge(color_img_ptr, color_img_buffer.get(), kernel, sigma_s, sigma_r, w, h, bpp);
	copy(color_img_buffer.get(), color_img_buffer.get()+w*h*bpp, color_img_ptr);
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[3]);
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	return 0;
}

