#include "global.h"
#include "cl_helper.h"
#include "bilateralfilter.h"
#include "utilities.h"
#include <IL/il.h>
#include <glog/logging.h>
#include <cmath>
#include <memory>
#include "timer.h"
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

	unique_ptr<float[]> weight_pixel_sum(new float[bpp]);
	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			float weight_sum = 0.0f;
			const float *cur_in = image_rgb_in.get()+y*w+x;
			float *cur_out = image_rgb_out.get()+y*w+x;
			for (int dy = -offset; dy <= offset; dy++) {
				for(int dx = -offset; dx <= offset; dx++) {
					const float *neighbor_in = cur_in+dy*w+dx;
					float spatial = exp( (dx*dx+dy*dy) * spacial_sigma_inverse);
					float range_diff = 0;
					for (int d = 0; d < bpp; ++d) {
						float diff_1d = cur_in[d*w*h] - neighbor_in[d*w*h];
						range_diff += diff_1d * diff_1d;
					}
					float range = exp(range_diff * color_sigma_inverse);
					float weight = range * spatial;
					weight_sum += weight;
					for (int d = 0; d < bpp; ++d) {
						weight_pixel_sum[d] += weight * neighbor_in[d*w*h];
					}
				}
			}
			for (int d = 0; d < bpp; ++d) {
				cur_out[d*w*h] = ClampToUint8((int)(weight_pixel_sum[d] / weight_sum + 0.5f));
				weight_pixel_sum[d] = 0;
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

void BilateralFilter::Run_ocl(const float *image_in, float *image_out)
{
	const int w = w_;
	const int h = h_;
	const int bpp = channel_;
	const int r = param_.radius;
	const float spacial_sigma = param_.spacial_sigma;
	const float color_sigma = param_.color_sigma;
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

	Clock tic, toc;
	long long elapsed;
	tic = GetNow();

	auto range_gaussian_table = GenerateGaussianTable(spacial_sigma, r+1);
	auto color_gaussian_table = GenerateGaussianTable(color_sigma, 256);
	cl_kernel kernel = device_manager->GetKernel("bilateral.cl", "bilateral");

	auto d_range_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, (r+1)*sizeof(float));
	auto d_color_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, 256*sizeof(float));
	auto d_in = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	device_manager->WriteMemory(range_gaussian_table.get(), *d_range_gaussian_table.get(), (r+1)*sizeof(float));
	device_manager->WriteMemory(color_gaussian_table.get(), *d_color_gaussian_table.get(), 256*sizeof(float));
	device_manager->WriteMemory(image_rgb_in.get(), *d_in.get(), w*h*bpp*sizeof(float));
	const int work_w = w-2*r;
	const int work_h = h-2*r;
	const size_t block_dim[2] = {32, 16};
	const size_t grid_dim[2] = {CeilDiv(work_w, 32)*32, CeilDiv(work_h, 16)*16};

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
			{&color_sigma, sizeof(float)},
			{d_range_gaussian_table.get(), sizeof(cl_mem)},
			{d_color_gaussian_table.get(), sizeof(cl_mem)}
		},
		2, grid_dim, nullptr, block_dim
	);
	
	toc = GetNow();
	elapsed = DiffUsInLongLong(tic, toc);
	LOG(INFO) << "Time: " << elapsed << "us";
	
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
