#include "gaussianfilter.h"
#include "enhancement.h"
#include "utilities.h"
#include "timer.h"
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
	auto image_size = w*h*bpp;
	LOG(INFO) << "Load image width = " << w << ", height = " << h;
	ILubyte *color_img_ptr = ilGetData();
	unique_ptr<float[]> original_float(new float[image_size]);
	unique_ptr<float[]> enhanced_float(new float[image_size]);
	IL_CHECK_ERROR();
	copy(color_img_ptr, color_img_ptr+image_size, original_float.get());

	float sigma;
	int radius;
	cout<<"<sigma> <radius> : ";
	cin>>sigma>>radius;

	// Filter
	Filter *filter;

	// Gaussian
	GaussianFilter gf;
	gf.SetParameter({sigma, radius});

	filter = dynamic_cast<Filter*>(&gf);
	filter->SetDimension(w, h, bpp);

	// Enhance
	Clock tic, toc;
	tic = GetNow();
	float weights[] = {2.0f, 2.0f};
	Enhance(original_float.get(), enhanced_float.get(), weights, w, h, bpp, sizeof(weights)/sizeof(float), filter);
	toc = GetNow();
	LOG(INFO) << "Time elapsed: " << DiffUsInLongLong(tic, toc) << "us";

	// Store enhanced image
	transform(enhanced_float.get(), enhanced_float.get()+image_size, color_img_ptr, [](const float x)->ILubyte {
		return ClampToUint8<int>(x);
	});
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[2]);
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	return 0;
}
