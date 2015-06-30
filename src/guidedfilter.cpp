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

	unique_ptr<float[]> a_r(new float[w*h*bpp]);
	unique_ptr<float[]> a_g(new float[w*h*bpp]);
	unique_ptr<float[]> a_b(new float[w*h*bpp]);
	unique_ptr<float[]> b(new float[w*h*bpp]);

	const int line_stride = bpp*w;

	unique_ptr<float[]> image_rgb_in(new float[w*h*bpp]);
	unique_ptr<float[]> I_rgb(new float[w*h*bpp]);
	unique_ptr<float[]> image_rgb_out(new float[w*h*bpp]);
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int d = 0; d < bpp; ++d) {
				image_rgb_in[y*w+x+d*w*h]=image_in[y*line_stride+x*bpp+d];
				I_rgb[y*w+x+d*w*h]=image_in[y*line_stride+x*bpp+d];
			}
		}
	}

	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			float sum_g[3]={};
			float square_g[3]={};
			float sum_in[3]={};
			float sum[3][3]={{0,0,0},{0,0,0},{0,0,0}};
			float corr_rg = 0;
			float corr_rb = 0;
			float corr_gb = 0;
			float mean_g[3]={};
			float squaremean_g[3]={};
			float mean_in[3]={};
			float var_I_rr = 0;
			float var_I_rg = 0;
			float var_I_rb = 0;
			float var_I_gg = 0;
			float var_I_gb = 0;
			float var_I_bb = 0;
			float a_temp[3][3]={}; 
			float b_temp[3]={};

			const int base = y*w + x;
			for (int d = 0; d < bpp; ++d) {
				for (int dy = -offset; dy <= offset; dy++) {
					for(int dx = -offset; dx <= offset; dx++) {
						const int neighbor_offset_pixel = dy*w+dx;
						const int neighbor_offset = dy*w+dx+d*w*h;
						const float pixel_I = I_rgb[base+neighbor_offset];
						const float pixel_in = image_rgb_in[base+neighbor_offset];

						sum_g[d] += pixel_I;
						square_g[d] += pixel_I * pixel_I;
						sum_in[d] += pixel_in; 
						sum[d][0] += I_rgb[base+neighbor_offset_pixel+0*w*h] * pixel_in;
						sum[d][1] += I_rgb[base+neighbor_offset_pixel+1*w*h] * pixel_in;
						sum[d][2] += I_rgb[base+neighbor_offset_pixel+2*w*h] * pixel_in;
						if (d==0) {
							corr_rg += I_rgb[base+neighbor_offset_pixel+0*w*h] * I_rgb[base+neighbor_offset_pixel+1*w*h];
							corr_rb += I_rgb[base+neighbor_offset_pixel+0*w*h] * I_rgb[base+neighbor_offset_pixel+2*w*h];
							corr_gb += I_rgb[base+neighbor_offset_pixel+1*w*h] * I_rgb[base+neighbor_offset_pixel+2*w*h];
						}
					}
				}
				mean_g[d] = sum_g[d] * inv_size;
				squaremean_g[d] = square_g[d] * inv_size;
				mean_in[d] = sum_in[d] * inv_size;
			}

			corr_rg *= inv_size; 
			corr_rb *= inv_size; 
			corr_gb *= inv_size; 
			var_I_rr = squaremean_g[0] - mean_g[0] * mean_g[0];
			var_I_rg = corr_rg - mean_g[0] * mean_g[1];
			var_I_rb = corr_rb - mean_g[0] * mean_g[2];
			var_I_gg = squaremean_g[1] - mean_g[1] * mean_g[1];
			var_I_gb = corr_gb - mean_g[1] * mean_g[2];
			var_I_bb = squaremean_g[2] - mean_g[2] * mean_g[2];

			float m[3][3];
			m[0][0] = var_I_rr + epsilon;
			m[0][1] = var_I_rg;
			m[0][2] = var_I_rb;
			m[1][0] = var_I_rg;
			m[1][1] = var_I_gg + epsilon;
			m[1][2] = var_I_gb;
			m[2][0] = var_I_rb;
			m[2][1] = var_I_gb;
			m[2][2] = var_I_bb + epsilon; 

			// computes the inverse of a matrix m
			float det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
			             m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
			             m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

			float invdet = 1 / det;
			float minv[3][3]; // inverse of matrix m
			minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
			minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
			minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
			minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
			minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
			minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
			minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
			minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
			minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;

			for (int d = 0; d < bpp; ++d) {
				a_temp[d][0]=minv[0][0]*(sum[d][0]-size*mean_g[0]*mean_in[d])+minv[0][1]*(sum[d][0]-size*mean_g[0]*mean_in[d])+minv[0][2]*(sum[d][0]-size*mean_g[0]*mean_in[d]);
				a_temp[d][1]=minv[1][0]*(sum[d][1]-size*mean_g[1]*mean_in[d])+minv[1][1]*(sum[d][1]-size*mean_g[1]*mean_in[d])+minv[1][2]*(sum[d][1]-size*mean_g[1]*mean_in[d]);
				a_temp[d][2]=minv[2][0]*(sum[d][2]-size*mean_g[2]*mean_in[d])+minv[2][1]*(sum[d][2]-size*mean_g[2]*mean_in[d])+minv[2][2]*(sum[d][2]-size*mean_g[2]*mean_in[d]);
				a_temp[d][0]*=inv_size;
				a_temp[d][1]*=inv_size;
				a_temp[d][2]*=inv_size;
			}
			float au = 0;
			for (int d = 0; d < bpp; ++d) {
				au = 0;
				au += a_temp[d][0]*mean_g[0]+a_temp[d][1]*mean_g[1]+a_temp[d][2]*mean_g[2];
				b_temp[d] = mean_in[d] - au; 
			}
			for (int d = 0; d < bpp; ++d) {
				a_r[base+d*w*h] = a_temp[d][0];
				a_g[base+d*w*h] = a_temp[d][1];
				a_b[base+d*w*h] = a_temp[d][2];
				b[base+d*w*h] = b_temp[d];
			}
		}
	}


	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			const int base = y*w + x;
			for (int d = 0; d < bpp; ++d) {
				int pixel_output = 0;
				float sum_a[3]={};
				float sum_b = 0;
				float mean_a[3]={};
				float mean_b = 0;
				for (int dx = -offset; dx <= offset; dx++) {
					for(int dy = -offset; dy <= offset; dy++) {
						const int neighbor_offset_pixel = dy*w+dx;
						const int neighbor_offset = dy*w+dx+d*w*h;
						const float pixel_I = I_rgb[base+neighbor_offset];
						const float pixel_in = image_rgb_in[base+neighbor_offset];
						sum_a[0] += a_r[base+neighbor_offset];
						sum_a[1] += a_g[base+neighbor_offset];
						sum_a[2] += a_b[base+neighbor_offset];
						sum_b += b[base+neighbor_offset];
					}
				}	
				mean_a[0] = sum_a[0] * inv_size;
				mean_a[1] = sum_a[1] * inv_size;
				mean_a[2] = sum_a[2] * inv_size;
				mean_b = sum_b * inv_size;
				pixel_output = mean_a[0]*I_rgb[base+0*w*h]+mean_a[1]*I_rgb[base+1*w*h]+mean_a[2]*I_rgb[base+2*w*h]+mean_b;
				image_rgb_out[base+d*w*h] = (pixel_output&0xffffff00)? ~(pixel_output>>24): pixel_output;
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

	unique_ptr<float[]> image_rgb_in(new float[w*h*bpp]);
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int d = 0; d < bpp; ++d) {
				image_rgb_in[y*w+x+d*w*h]=image_in[y*line_stride+x*bpp+d];
			}
		}
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
	device_manager->WriteMemory(image_rgb_in.get(), *d_in.get(), w*h*bpp*sizeof(float));
	device_manager->WriteMemory(image_rgb_in.get(), *d_I.get(), w*h*bpp*sizeof(float));

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
