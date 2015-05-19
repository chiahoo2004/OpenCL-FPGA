#include "guidedfilter.h"
#include <iostream>
#include <memory>
#include <IL/il.h>
#include <glog/logging.h>
using namespace std;

#define IL_CHECK_ERROR() {auto e = ilGetError();CHECK_EQ(e, IL_NO_ERROR)<<e;}
int main(int argc, char const* argv[])
{
	// Initialize Glog and libIL
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

	int windowSize, epsilon;
	cout<<"<window size> <epsilon> : ";
	cin>>windowSize>>epsilon;
	vector<vector<vector<double> > > a, b;
	a.resize(w);
	b.resize(w);
	for(int i=0; i<w; ++i) { 
		a[i].resize(h);
		b[i].resize(h);	
		for(int j=0; j<h; ++j) { 
			a[i][j].resize(bpp);
			b[i][j].resize(bpp);	
		}	
	}

	// Guided filter
	GuidedFilter gf;
	gf.Run(color_img_ptr, color_img_buffer.get(), color_img_ptr, a, b, windowSize, epsilon, w, h, bpp);

	// store image
	copy(color_img_buffer.get(), color_img_buffer.get()+w*h*bpp, color_img_ptr);
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[2]);
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	return 0;
}
