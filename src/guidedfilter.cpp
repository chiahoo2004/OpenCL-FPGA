#include "global.h"
#include "cl_helper.h"
#include "guidedfilter.h"
#include "utilities.h"
#include <IL/il.h>
#include <glog/logging.h>
#include <cmath>
using namespace std;

void GuidedFilter::SetDimension(const int w, const int h, const int channel)
{
	Filter::SetDimension(w, h, channel);
	a.reset(new float[w*h*channel]);
	b.reset(new float[w*h*channel]);
}

void GuidedFilter::Run_cxx(const float *image_in, float* image_out)
{
	const int bpp = channel_;
	const int w = w_;
	const int h = h_;
	const float epsilon = param_.epsilon;
	const int radius = param_.radius;
	const float* I = image_in;
	const int length = radius*2+1;
	const int offset = radius;
	const int size = length * length;
	const float inv_size = 1.0/size;
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";

	const int line_stride = bpp*w;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			const int base = y*line_stride + x*bpp;
			for (int d = 0; d < bpp; ++d) {
				float sum_g = 0;
				float square_g = 0;
				float sum_in = 0;
				float sum = 0;
				for (int dy = -offset; dy <= offset; dy++) {
					for(int dx = -offset; dx <= offset; dx++) {
						const int neighbor_offset = dy*line_stride+dx*bpp+d;
						const float pixel_I = I[base+neighbor_offset];
						const float pixel_in = image_in[base+neighbor_offset];
						sum_g += pixel_I;
						square_g += pixel_I * pixel_I;
						sum_in += pixel_in;
						sum += pixel_in * pixel_I;
					}
				}

				float mean_g = sum_g * inv_size;
				float squaremean_g = square_g * inv_size;
				float mean_in = sum_in * inv_size;
				float variance_g = squaremean_g - mean_g * mean_g;

				float a_temp = (sum*inv_size - mean_g*mean_in) / (variance_g+epsilon);
				float b_temp = mean_in - a_temp*mean_g;
				a[base+d] = a_temp;
				b[base+d] = b_temp;
			}
		}
	}

	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			const int base = y*line_stride + x*bpp;
			for (int d = 0; d < bpp; ++d) {
				float sum_a = 0;
				float sum_b = 0;
				for (int dy = -offset; dy <= offset; dy++) {
					for(int dx = -offset; dx <= offset; dx++) {
						sum_a += a[base+dy*line_stride+dx*bpp+d];
						sum_b += b[base+dy*line_stride+dx*bpp+d];
					}
				}
				image_out[base+d] = (sum_a*I[base+d]+sum_b) * inv_size;
			}
		}
	}
}

void GuidedFilter::Run_ocl(const float *image_in, float* image_out)
{
	const int w = w_;
	const int h = h_;
	const int bpp = channel_;
	const int r = param_.radius;
	const float epsilon = param_.epsilon;
	const float* I = image_in;
	const int length = r*2+1;
	const int offset = r;
	const int size = length * length;
	const int line_stride = w*bpp;
	if (w <= 2*r || h <= 2*r) {
		LOG(WARNING) << "No work to do";
		return;
	}
//	auto range_gaussian_table = GenerateGaussianTable(spacial_sigma, r+1);
	cl_kernel kernel1 = device_manager->GetKernel("guided.cl", "guided");

	auto d_a_r = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_a_g = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_a_b = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_b = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_I = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));

//	auto d_range_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, (r+1)*sizeof(float));
	auto d_in = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
	auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
//	device_manager->WriteMemory(range_gaussian_table.get(), *d_range_gaussian_table.get(), (r+1)*sizeof(float));
	device_manager->WriteMemory(image_in, *d_in.get(), w*h*bpp*sizeof(float));
	device_manager->WriteMemory(image_in, *d_I.get(), w*h*bpp*sizeof(float));

	const int work_w = w-2*r;
	const int work_h = h-2*r;
	const size_t block_dim[2] = {32, 16};
	const size_t grid_dim[2] = {CeilDiv(w, block_dim[0])*block_dim[0], CeilDiv(h, block_dim[1])*block_dim[1]};

	device_manager->Call(
		kernel1,
		{
			{d_in.get(), sizeof(cl_mem)},
			{d_out.get(), sizeof(cl_mem)},
			{&r, sizeof(int)},
			{&epsilon, sizeof(float)},
			{&work_w, sizeof(int)},
			{&work_h, sizeof(int)},
			{&bpp, sizeof(int)},
			{&line_stride, sizeof(int)},
			{d_a_r.get(), sizeof(cl_mem)},
			{d_a_g.get(), sizeof(cl_mem)},
			{d_a_b.get(), sizeof(cl_mem)},
			{d_b.get(), sizeof(cl_mem)},
			{d_I.get(), sizeof(cl_mem)}
		},
		2, grid_dim, nullptr, block_dim
	);

	cl_kernel kernel2 = device_manager->GetKernel("guidedtwo.cl", "guidedtwo");

	device_manager->Call(
		kernel2,
		{
			{d_in.get(), sizeof(cl_mem)},
			{d_out.get(), sizeof(cl_mem)},
			{&r, sizeof(int)},
			{&epsilon, sizeof(float)},
			{&work_w, sizeof(int)},
			{&work_h, sizeof(int)},
			{&bpp, sizeof(int)},
			{&line_stride, sizeof(int)},
			{d_a_r.get(), sizeof(cl_mem)},
			{d_a_g.get(), sizeof(cl_mem)},
			{d_a_b.get(), sizeof(cl_mem)},
			{d_b.get(), sizeof(cl_mem)},
			{d_I.get(), sizeof(cl_mem)}
		},
		2, grid_dim, nullptr, block_dim
	);

	device_manager->ReadMemory(image_out, *d_out.get(), w*h*bpp*sizeof(float));
//	device_manager->ReadMemory(image_out, *d_in.get(), w*h*bpp*sizeof(float));
}
