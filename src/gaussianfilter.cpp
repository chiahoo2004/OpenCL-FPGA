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

	auto range_gaussian_table = GenerateGaussianTable(param_.spacial_sigma, 2*radius+1);
	unique_ptr<float[]> mid(new float[w*h*bpp]);
	const int line_stride = bpp*w;

	unique_ptr<float[]> image_rgb_in(new float[w*h*bpp]);
	unique_ptr<float[]> image_rgb_out(new float[w*h*bpp]);
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int d = 0; d < bpp; ++d) {
				image_rgb_in[y*w+x+d*w*h]=image_in[y*line_stride+x*bpp+d];
			}
		}
	}

	for (int y = radius; y < h-radius; ++y) {
		for (int x = radius; x < w-radius; ++x) {
			for (int d = 0; d < bpp; ++d) {
				float weight_sum = 0.0f;
				float weight_pixel_sum = 0.0f;
				
				for (int i = -radius; i <= radius; ++i) {
					int range_diff = abs(i);
					weight_sum += range_gaussian_table[range_diff];
					weight_pixel_sum += range_gaussian_table[range_diff] * image_rgb_in[(y+i)*w+x+d*w*h];
				}

				const int mid_output = weight_pixel_sum/weight_sum + 0.5f;
				mid[x*w+y+d*w*h] = ((int)mid_output&0xffffff00)? ~((int)mid_output>>24): (int)mid_output;
			}
		}
	}
	
	for (int y = radius; y < h-radius; ++y) {
		for (int x = radius; x < w-radius; ++x) {
			for (int d = 0; d < bpp; ++d) {
				float weight_sum = 0.0f;
				float weight_pixel_sum = 0.0f;
				
				for (int i = -radius; i <= radius; ++i) {
					int range_diff = abs(i);
					weight_sum += range_gaussian_table[range_diff];
					weight_pixel_sum += range_gaussian_table[range_diff] * mid[(y+i)*w+x+d*w*h];
				}

				const int mid_output = weight_pixel_sum/weight_sum + 0.5f;
				image_rgb_out[x*w+y+d*w*h] = ((int)mid_output&0xffffff00)? ~((int)mid_output>>24): (int)mid_output;
			}
		}
	}
	
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int d = 0; d < bpp; ++d) {
				image_out[y*line_stride+x*bpp+d]=image_rgb_out[y*w+x+d*w*h];
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

	unique_ptr<float[]> image_rgb_in(new float[w*h*bpp]);
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int d = 0; d < bpp; ++d) {
				image_rgb_in[y*w+x+d*w*h]=image_in[y*line_stride+x*bpp+d];
			}
		}
	}

	auto range_gaussian_table = GenerateGaussianTable(spacial_sigma, 2*r+1);
	cl_kernel kernel = device_manager->GetKernel("gaussian1d.cl", "gaussian1d");

	auto d_mid = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));

	auto d_range_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, (2*r+1)*sizeof(float));
	auto d_in = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	device_manager->WriteMemory(range_gaussian_table.get(), *d_range_gaussian_table.get(), (2*r+1)*sizeof(float));
	device_manager->WriteMemory(image_rgb_in.get(), *d_in.get(), w*h*bpp*sizeof(float));

	const int work_w = w-2*r;
	const int work_h = h-2*r;
	const size_t block_dim[2] = {32, 16};
	const size_t grid_dim[2] = {CeilDiv(work_w, 32)*32, CeilDiv(work_h, 16)*16};

	device_manager->Call(
		kernel,
		{
			{d_in.get(), sizeof(cl_mem)},
			{d_mid.get(), sizeof(cl_mem)},
			{&r, sizeof(int)},
			{&work_w, sizeof(int)},
			{&work_h, sizeof(int)},
			{&bpp, sizeof(int)},
			{&line_stride, sizeof(int)},
			{d_range_gaussian_table.get(), sizeof(cl_mem)}
		},
		2, grid_dim, nullptr, block_dim
	);

	float* mid = new float[w*h*bpp];
	device_manager->ReadMemory(mid, *d_mid.get(), w*h*bpp*sizeof(float));

	device_manager->Call(
		kernel,
		{
			{d_mid.get(), sizeof(cl_mem)},
			{d_out.get(), sizeof(cl_mem)},
			{&r, sizeof(int)},
			{&work_w, sizeof(int)},
			{&work_h, sizeof(int)},
			{&bpp, sizeof(int)},
			{&line_stride, sizeof(int)},
			{d_range_gaussian_table.get(), sizeof(cl_mem)}
		},
		2, grid_dim, nullptr, block_dim
	);

	unique_ptr<float[]> image_rgb_out(new float[w*h*bpp]);
	device_manager->ReadMemory(image_rgb_out.get(), *d_out.get(), w*h*bpp*sizeof(float));	
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int d = 0; d < bpp; ++d) {
				image_out[y*line_stride+x*bpp+d]=image_rgb_out[y*w+x+d*w*h];
			}
		}
	}

}
