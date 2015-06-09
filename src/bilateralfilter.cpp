#include "global.h"
#include "cl_helper.h"
#include "bilateralfilter.h"
#include "utilities.h"
#include <IL/il.h>
#include <glog/logging.h>
#include <cmath>
#include <memory>
using namespace std;

void BilateralFilter::SetDimension(const int w, const int h, const int channel)
{
	Filter::SetDimension(w, h, channel);
}

void BilateralFilter::Run_cxx(const float *image_in, float *image_out)
{
	const int bpp = channel_;
	const int w = w_;
	const int h = h_;
	const int radius = param_.radius;
	const float spacial_sigma = param_.spacial_sigma;
	const float color_sigma = param_.color_sigma;
	const float spacial_sigma_inverse = -1.0f / (2.0f * spacial_sigma * spacial_sigma);
	const float color_sigma_inverse = -1.0f / (2.0f * color_sigma * color_sigma);
	const int size = radius*2+1;
	const int offset = radius;
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";
	
	const int line_stride = bpp*w;LOG(INFO)<<(h-offset-1)*line_stride+line_stride-bpp*offset-1;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = bpp*offset; x < line_stride-bpp*offset; ++x) {

			float image = 0.0f;
			float weight_sum = 0.0f;
			float weight_pixel_sum = 0.0f;
			for (int a = -offset; a <= offset; a++) {
				for(int b = -offset; b <= offset; b++) {
					float spatial = exp(-(x*x+y*y) * spacial_sigma_inverse);
					float range_diff = image_in[x+y*line_stride]-image_in[x+b*bpp+(y+a)*line_stride];
					float range = exp(-range_diff * range_diff * color_sigma_inverse);
					float weight = range * spatial;
					weight_sum += weight;
					weight_pixel_sum += weight * image_in[(y+a)*line_stride+(x+b*bpp)];
				}
			}
			image_out[y*line_stride+x] = ClampToUint8(int(weight_pixel_sum/weight_sum + 0.5f));
		}
	}
}

void BilateralFilter::Run_ocl(const float *image_in, float *image_out)
{
	const int w = w_;
	const int h = h_;
	const int bpp = channel_;
	const int r = param_.radius;
	const float spacial_sigma = param_.spacial_sigma;
	const float color_sigma = param_.color_sigma;
	if (w <= 2*r || h <= 2*r) {
		LOG(WARNING) << "No work to do";
		return;
	}
	auto range_gaussian_table = GenerateGaussianTable(spacial_sigma, r+1);
	auto color_gaussian_table = GenerateGaussianTable(color_sigma, 256);
	cl_kernel kernel = device_manager->GetKernel("bilateral.cl", "bilateral");

	auto d_range_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, (r+1)*sizeof(float));
	auto d_color_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, 256*sizeof(float));
	auto d_in = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	device_manager->WriteMemory(range_gaussian_table.get(), *d_range_gaussian_table.get(), (r+1)*sizeof(float));
	device_manager->WriteMemory(color_gaussian_table.get(), *d_color_gaussian_table.get(), 256*sizeof(float));
	device_manager->WriteMemory(image_in, *d_in.get(), w*h*bpp*sizeof(float));
	const int work_w = w-2*r;
	const int work_h = h-2*r;
	const size_t block_dim[3] = {32, 16, 1};
	const size_t grid_dim[3] = {CeilDiv(work_w, 32)*32, CeilDiv(work_h, 16)*16, bpp};

	device_manager->Call(
		kernel,
		{
			{d_in.get(), sizeof(cl_mem)},
			{d_out.get(), sizeof(cl_mem)},
			{&r, sizeof(int)},
			{&work_w, sizeof(int)},
			{&work_h, sizeof(int)},
			{&bpp, sizeof(int)},
			{&w, sizeof(int)},
			{d_range_gaussian_table.get(), sizeof(cl_mem)},
			{d_color_gaussian_table.get(), sizeof(cl_mem)}
		},
		3, grid_dim, nullptr, block_dim
	);
	device_manager->ReadMemory(image_out, *d_out.get(), w*h*bpp*sizeof(float));
}
