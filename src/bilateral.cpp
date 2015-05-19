#include "bilateralfilter.h"
#include "enhancement.h"
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
	unique_ptr<ILubyte[]> color_img_buffer2(new ILubyte[w*h*bpp]);
	unique_ptr<ILubyte[]> out(new ILubyte[w*h*bpp]);
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

	// store image
	int layer = 3;
	ILubyte** images_clear_to_blur = new ILubyte*[layer];
	for(int i = 0; i < layer ; ++i){
		images_clear_to_blur[i] = new ILubyte[w*h*bpp];
	}
	copy(color_img_ptr, color_img_ptr+w*h*bpp, *images_clear_to_blur);
	copy(color_img_buffer.get(), color_img_buffer.get()+w*h*bpp, color_img_ptr);

	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[2]);
	IL_CHECK_ERROR();

	unique_ptr<float[]> weights(new float[layer]);
  	*(weights.get()) = 1;
  	cout<<"weights <original-first smooth> <first smooth-second smooth> : ";
  	cin>>*(weights.get()+1)>>*(weights.get()+2);
  	bf.Run(color_img_buffer.get(), color_img_buffer2.get(), kernel, sigma_s, sigma_r, w, h, bpp);

	copy(color_img_buffer.get(), color_img_buffer.get()+w*h*bpp, *(images_clear_to_blur+1));
	copy(color_img_buffer2.get(), color_img_buffer2.get()+w*h*bpp, *(images_clear_to_blur+2));
	Enhance((unsigned char**)images_clear_to_blur, out.get(), weights.get(), w, h, bpp, layer);
	copy(out.get(), out.get()+w*h*bpp, color_img_ptr);

	for(int i = 0; i < layer ; ++i){
		delete [] images_clear_to_blur[i];
	}
	delete [] images_clear_to_blur;

	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[3]);
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	return 0;
}

