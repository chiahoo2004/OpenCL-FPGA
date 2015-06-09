#include "global.h"
#include "cl_helper.h"
#include "gaussianfilter.h"
#include "utilities.h"
#include <IL/il.h>
#include <glog/logging.h>
#include <cmath>
#include <memory>
using namespace std;

void GaussianFilter::SetDimension(const int w, const int h, const int channel)
{
	Filter::SetDimension(w, h, channel);
}

void GaussianFilter::Run_cxx(const float *image_in, float* image_out)
{
	const int w = w_;
	const int h = h_;
	const int radius = param_.radius;
	const int bpp = channel_;
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";

	auto kernel = CreateGaussianKernel(param_.spacial_sigma, radius);
	int buf_size = radius*2+1;

	unique_ptr<float[]> mid(new float[w*h*bpp]);

	const int line_stride = bpp*w;
	for (int y = radius; y < h-radius; ++y) {
		for (int a = 0; a < bpp; ++a){
			for (int b = 0; b < 2*radius+1; ++b){
				for (int c = -radius; c < radius; ++c){
					mid[y*line_stride+b*bpp+a] += kernel[c+radius] * image_in[(y+c)*line_stride+(a+bpp*b)];;
				}
			}
		}
		for (int x = (radius+1); x < w-radius; ++x) {
			for (int d = 0; d < bpp; ++d) {
				for (int c = -radius; c <= radius; ++c) {
					mid[y*line_stride+(x+radius)*bpp+d] += kernel[c+radius] * image_in[(y+c)*line_stride+(d+bpp*(x+radius))];
				}

				float pixel_output = 0;
				for (int c = -radius; c < radius; ++c)
				{
					pixel_output += kernel[c+radius] * mid[y*line_stride+(x+c)*bpp+d];
				}
				image_out[y*line_stride+x*bpp+d] = pixel_output;
			}
		}
	}
}

void GaussianFilter::Run_ocl(const float *image_in, float* image_out)
{
	const int w = w_;
	const int h = h_;
	const int bpp = channel_;
	const int r = param_.radius;
	const float spacial_sigma = param_.spacial_sigma;
	const int line_stride = w*bpp;
	if (w <= 2*r || h <= 2*r) {
		LOG(WARNING) << "No work to do";
		return;
	}
	auto range_gaussian_table = GenerateGaussianTable(spacial_sigma, 2*r+1);
	cl_kernel kernel = device_manager->GetKernel("gaussian1d.cl", "gaussian1d");

	auto d_mid = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));

	auto d_range_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, (2*r+1)*sizeof(float));
	auto d_in = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	device_manager->WriteMemory(range_gaussian_table.get(), *d_range_gaussian_table.get(), (2*r+1)*sizeof(float));
	device_manager->WriteMemory(image_in, *d_in.get(), w*h*bpp*sizeof(float));

	const int work_w = w-2*r;
	const int work_h = h-2*r;
	const size_t block_dim[3] = {32, 16, 1};
	const size_t grid_dim[3] = {w, h, bpp};

	device_manager->Call(
		kernel,
		{
			{d_in.get(), sizeof(cl_mem)},
			{d_out.get(), sizeof(cl_mem)},
			{&r, sizeof(int)},
			{&work_w, sizeof(int)},
			{&work_h, sizeof(int)},
			{&bpp, sizeof(int)},
			{&line_stride, sizeof(int)},
			{d_range_gaussian_table.get(), sizeof(cl_mem)},
			{d_mid.get(), sizeof(cl_mem)}
		},
		3, grid_dim, nullptr, block_dim
	);

	float* mid = new float[w*h*bpp];
	device_manager->ReadMemory(mid, *d_mid.get(), w*h*bpp*sizeof(float));

	cl_kernel kernel2 = device_manager->GetKernel("gaussian1dtwo.cl", "gaussian1dtwo");

	device_manager->Call(
		kernel2,
		{
			{d_in.get(), sizeof(cl_mem)},
			{d_out.get(), sizeof(cl_mem)},
			{&r, sizeof(int)},
			{&work_w, sizeof(int)},
			{&work_h, sizeof(int)},
			{&bpp, sizeof(int)},
			{&line_stride, sizeof(int)},
			{d_range_gaussian_table.get(), sizeof(cl_mem)},
			{d_mid.get(), sizeof(cl_mem)}
		},
		3, grid_dim, nullptr, block_dim
	);

	device_manager->ReadMemory(image_out, *d_out.get(), w*h*bpp*sizeof(float));

}
