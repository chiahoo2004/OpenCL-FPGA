#include "cl_helper.h"
#include "global.h"
#include "gaussianfilter.h"
#include "bilateralfilter.h"
#include "guidedfilter.h"
#include "enhancement.h"
#include "utilities.h"
#include "timer.h"
#include <algorithm>
#include <memory>
#include <IL/il.h>
#include <glog/logging.h>
#include <CL/cl.h>
#include <chrono>
#include <cmath>
#include <cstdint>
using namespace std;
using namespace google;
using namespace std::chrono;
DeviceManager *device_manager;
enum FilterMethod {
	Gaussian, Bilateral, Guided
};

void InitOpenCL(size_t id)
{
	// platforms
	auto platforms = GetPlatforms();
	LOG(INFO) << platforms.size() << " platform(s) found";
	int last_nvidia_platform = -1;
	for (size_t i = 0; i < platforms.size(); ++i) {
		auto platform_name = GetPlatformName(platforms[i]);
		LOG(INFO) << ">>> Name: " << platform_name.data();
		if (strcmp("NVIDIA CUDA", platform_name.data()) == 0) {
			last_nvidia_platform = i;
		}
	}
	CHECK_NE(last_nvidia_platform, -1) << "Cannot find any NVIDIA CUDA platform";

	// devices under the last CUDA platform
	auto devices = GetPlatformDevices(platforms[last_nvidia_platform]);
	LOG(INFO) << devices.size() << " device(s) found under some platform";
	for (size_t i = 0; i < devices.size(); ++i) {
		auto device_name = GetDeviceName(devices[i]);
		LOG(INFO) << ">>> Name: " << device_name.data();
	}
	CHECK_LT(id, devices.size()) << "Cannot find device " << id;
	device_manager = new DeviceManager(devices[id]);
}

#define IL_CHECK_ERROR() {auto e = ilGetError();CHECK_EQ(e, IL_NO_ERROR)<<e;}
int main(int argc, char** argv)
{
	// Initialize Glog and libIL
	InitGoogleLogging(argv[0]);
	FLAGS_logtostderr = true;
	CHECK_EQ(argc, 3) << "Usage: <executable> <input> <output>";
	ilInit();
	LOG(INFO) << "Using devil library version " << ilGetInteger(IL_VERSION_NUM);
	InitOpenCL(0);

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

	// Filter, fix it at compile time now
	Filter *filter;
//	FilterMethod filter_method = FilterMethod::Bilateral;
	FilterMethod filter_method = FilterMethod::Gaussian;
//	FilterMethod filter_method = FilterMethod::Guided;
	switch (filter_method) {
		case FilterMethod::Gaussian: {
			GaussianFilter *gf = new GaussianFilter;
			gf->SetParameter({10.0f, 1});
			filter = dynamic_cast<Filter*>(gf);
			break;
		}
		case FilterMethod::Bilateral: {
			BilateralFilter *bf = new BilateralFilter;
			bf->SetParameter({30.0f, 30.0f, 5});
			filter = dynamic_cast<Filter*>(bf);
			break;
		}
		case FilterMethod::Guided: {
			GuidedFilter *gf = new GuidedFilter;
			gf->SetParameter({500.0f, 1});
			filter = dynamic_cast<Filter*>(gf);
			break;
		}
	}
	filter->SetDimension(w, h, bpp); // move to Enhance?

	// Let's run the code
	Clock tic, toc;
	long long elapsed_cxx, elapsed_ocl;

	// C++
	tic = GetNow();
	filter->Run_cxx(original_float.get(), enhanced_float.get());
	toc = GetNow();
	elapsed_cxx = DiffUsInLongLong(tic, toc);
	
	// OpenCL
//	device_manager->GetKernel("bilateral.cl", "bilateral"); // preload the kernel
	device_manager->GetKernel("gaussian1d.cl", "gaussian1d"); // preload the kernel
//	device_manager->GetKernel("gaussian1dtwo.cl", "gaussian1dtwo");
//	device_manager->GetKernel("guided.cl", "guided");
//	device_manager->GetKernel("guidedtwo.cl", "guidedtwo");
	tic = GetNow();
	filter->Run_ocl(original_float.get(), enhanced_float.get());
	toc = GetNow();
	elapsed_ocl = DiffUsInLongLong(tic, toc);

	LOG(INFO) << "Without OpenCL: " << elapsed_cxx << "us";
	LOG(INFO) << "With OpenCL: " << elapsed_ocl << "us";
	LOG(INFO) << "Speedup: " << elapsed_cxx/float(elapsed_ocl) << "x";
/*
	tic = GetNow();
	float weights[] = {2.0f, 2.0f};
	Enhance(original_float.get(), enhanced_float.get(), weights, w, h, bpp, sizeof(weights)/sizeof(float), filter);
	toc = GetNow();
	LOG(INFO) << "Time elapsed: " << DiffUsInLongLong(tic, toc) << "us";
*/
	// Store enhanced image
	transform(enhanced_float.get(), enhanced_float.get()+image_size, color_img_ptr, [](const float x)->ILubyte {
		return ClampToUint8<int>(x);
	});
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(argv[2]);
	IL_CHECK_ERROR();

	ilDeleteImages(1, &image);
	delete device_manager;
	return 0;
}
