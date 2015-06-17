#include "global.h"
#include "cl_helper.h"
#include "guidedfilter.h"
#include "utilities.h"
#include <IL/il.h>
#include <glog/logging.h>
#include <cmath>
#include <memory>
using namespace std;

//#define DEBUG1 1  
//#define DEBUG2 1  

void GuidedFilter::SetDimension(const int w, const int h, const int channel)
{
	Filter::SetDimension(w, h, channel);
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
	CHECK_NE(w, 0) << "Width might not be 0";
	CHECK_NE(h, 0) << "Height might not be 0";
	
	unique_ptr<float[]> a(new float[w*h*bpp]);
	unique_ptr<float[]> b(new float[w*h*bpp]);

	const int line_stride = bpp*w;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			for (int d = 0; d < bpp; ++d) {
				float sum_g = 0;
				float square_g = 0;
				float sum_in = 0;
				float sum = 0;
				
				float mean_g, squaremean_g, mean_in, variance_g, a_temp, b_temp;

				for (int a = -offset; a <= offset; a++) {
					for(int b = -offset; b <= offset; b++) {

						sum_g += I[(y+a)*line_stride+(x*bpp+d+b*bpp)];
						
						square_g += I[(y+a)*line_stride+(x*bpp+d+b*bpp)] * I[(y+a)*line_stride+(x*bpp+d+b*bpp)];
						sum_in += image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)]; 
						
						sum += I[(y+a)*line_stride+(x*bpp+d+b*bpp)] * image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)];

						#ifdef DEBUG1
						DLOG(INFO)<<"I["<<(y+a)*line_stride+(x*bpp+d+b*bpp)<<"] = "<<I[(y+a)*line_stride+(x*bpp+d+b*bpp)]<<endl;
						DLOG(INFO)<<"image["<<(y+a)*line_stride+(x*bpp+d+b*bpp)<<"] = "<<image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)]<<endl;
						#endif
					}
				}

				mean_g = sum_g/size;
				squaremean_g = square_g/size;
				mean_in = sum_in/size;
				variance_g = squaremean_g - mean_g * mean_g;
				

				a_temp = (sum - size*mean_g*mean_in) / (size*(variance_g+epsilon));
				b_temp = mean_in - a_temp*mean_g;


				#ifdef DEBUG1
				DLOG(INFO)<<"mean_g = "<<mean_g<<endl;
				DLOG(INFO)<<"squaremean_g = "<<squaremean_g<<endl;
				DLOG(INFO)<<"mean_in = "<<mean_in<<endl;
				DLOG(INFO)<<"variance_g = "<<variance_g<<endl;
				DLOG(INFO)<<"sum = "<<sum<<endl;
				DLOG(INFO)<<"a_temp = "<<a_temp<<endl;
				DLOG(INFO)<<"b_temp = "<<b_temp<<endl;
				#endif
			
				a[x*bpp+y*line_stride+d] = a_temp;
				b[x*bpp+y*line_stride+d] = b_temp;
				#ifdef DEBUG1
				DLOG(INFO)<<"a["<<x<<"]["<<y<<"]["<<d<<"] = "<<a_temp<<endl;
				DLOG(INFO)<<"b["<<x<<"]["<<y<<"]["<<d<<"] = "<<b_temp<<endl;
				#endif

				#ifdef DEBUG1
				DLOG(INFO)<<"pause"<<endl;
				fgetc(stdin);
				#endif
			}
		}
	}

	int pixel_output = 0;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			for (int d = 0; d < bpp; ++d) {
				float sum_a = 0;
				float sum_b = 0;
				float mean_a, mean_b;

				for (int i = -offset; i <= offset; i++) {
					for(int j = -offset; j <= offset; j++) {
						sum_a += a[(x+i)*bpp+(y+j)*line_stride+d];
						sum_b += b[(x+i)*bpp+(y+j)*line_stride+d];
						#ifdef DEBUG2
						DLOG(INFO)<<"sum_a += "<<sum_a<<endl;
						DLOG(INFO)<<"sum_b += "<<sum_b<<endl;
						#endif
					}
				}	

				mean_a = sum_a / size;
				mean_b = sum_b / size;
				#ifdef DEBUG2
				DLOG(INFO)<<"mean_a["<<y*line_stride+x*bpp+d<<"] = "<<mean_a<<endl;
				DLOG(INFO)<<"mean_b["<<y*line_stride+x*bpp+d<<"] = "<<mean_b<<endl;
				#endif

				pixel_output = mean_a*I[y*line_stride+x*bpp+d] +mean_b;
				image_out[y*line_stride+x*bpp+d] = ClampToUint8(abs(pixel_output)*1);
				#ifdef DEBUG2
				if(y*line_stride+x*bpp+d>100000)
				DLOG(INFO)<<"x:"<<x<<"   y:"<<y;
				DLOG(INFO)<<"pixel_output["<<y*line_stride+x*bpp+d<<"] ("<<image_out[y*line_stride+x*bpp+d]<<") = mean_a ("<<mean_a<<")   *   I["<<y*line_stride+x*bpp+d<<"] ("<<I[y*line_stride+x*bpp+d]<<")   +   mean_b ("<<mean_b<<")"<<endl;
				DLOG(INFO)<<"pause"<<endl;
				#endif
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












































		int x=1;
		int y=1;



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

		unique_ptr<float[]> a_r(new float[w*h*bpp]);
		unique_ptr<float[]> a_g(new float[w*h*bpp]);
		unique_ptr<float[]> a_b(new float[w*h*bpp]);
		unique_ptr<float[]> b(new float[w*h*bpp]);


		for (int d = 0; d < bpp; ++d) {

			for (int a = -offset; a <= offset; a++) {
				for(int b = -offset; b <= offset; b++) {

					sum_g[d] += I[(y+a)*line_stride+(x*bpp+d+b*bpp)];
					
					square_g[d] += I[(y+a)*line_stride+(x*bpp+d+b*bpp)] * I[(y+a)*line_stride+(x*bpp+d+b*bpp)];
					sum_in[d] += image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)]; 
					
					sum[d][0] += I[(y+a)*line_stride+(x*bpp+ 0 +b*bpp)] * image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)];
					sum[d][1] += I[(y+a)*line_stride+(x*bpp+ 1 +b*bpp)] * image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)];
					sum[d][2] += I[(y+a)*line_stride+(x*bpp+ 2 +b*bpp)] * image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)];

					if (d==0) {
						corr_rg += I[(y+a)*line_stride+(x*bpp+ 0 +b*bpp)] * I[(y+a)*line_stride+(x*bpp+ 1 +b*bpp)];
						corr_rb += I[(y+a)*line_stride+(x*bpp+ 0 +b*bpp)] * I[(y+a)*line_stride+(x*bpp+ 2 +b*bpp)];
						corr_gb += I[(y+a)*line_stride+(x*bpp+ 1 +b*bpp)] * I[(y+a)*line_stride+(x*bpp+ 2 +b*bpp)];
					}

					#ifdef DEBUG1
					DLOG(INFO)<<"I["<<(y+a)*line_stride+(x*bpp+d+b*bpp)<<"] = "<<I[(y+a)*line_stride+(x*bpp+d+b*bpp)]<<endl;
					DLOG(INFO)<<"image["<<(y+a)*line_stride+(x*bpp+d+b*bpp)<<"] = "<<in[(y+a)*line_stride+(x*bpp+d+b*bpp)]<<endl;
					#endif

				}
			}

			mean_g[d] = sum_g[d]/size;
			squaremean_g[d] = square_g[d]/size;
			mean_in[d] = sum_in[d]/size;

		}


/*
		for (int d = 0; d < bpp; ++d) {
			for (int a = -offset; a <= offset; a++) {
					for(int b = -offset; b <= offset; b++) {
						LOG(INFO)<<"image_in["<<y+a<<"]["<<x+b<<"]["<<d<<"]="<<image_in[(y+a)*line_stride+(x*bpp+d+b*bpp)];
				}
			}
		}
*/

/*
		for (int d = 0; d < bpp; ++d) {
			LOG(INFO)<<mean_g[d];
			LOG(INFO)<<squaremean_g[d];
			LOG(INFO)<<mean_in[d];
		}
*/
		corr_rg /= size; 
		corr_rb /= size; 
		corr_gb /= size; 

		LOG(INFO)<<corr_rg;
		LOG(INFO)<<mean_g[0];
		LOG(INFO)<<mean_g[1];

		var_I_rr = squaremean_g[0] - mean_g[0] * mean_g[0];
		var_I_rg = corr_rg - mean_g[0] * mean_g[1];
		var_I_rb = corr_rb - mean_g[0] * mean_g[2];
		var_I_gg = squaremean_g[1] - mean_g[1] * mean_g[1];
		var_I_gb = corr_gb - mean_g[1] * mean_g[2];
		var_I_bb = squaremean_g[2] - mean_g[2] * mean_g[2];


		LOG(INFO)<<var_I_rr;
		LOG(INFO)<<var_I_rg;
		LOG(INFO)<<var_I_rb;
		LOG(INFO)<<var_I_gg;
		LOG(INFO)<<var_I_gb;
		LOG(INFO)<<var_I_bb;



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


		for (int a = 0; a < bpp; a++) {
			LOG(INFO)<<m[a][0]<<" "<<m[a][1]<<" "<<m[a][2];
		}


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


		for (int a = 0; a < bpp; a++) {
			LOG(INFO)<<minv[a][0]<<" "<<minv[a][1]<<" "<<minv[a][2];
		}


		for (int d = 0; d < bpp; ++d) {


			a_temp[d][0]=minv[0][0]*(sum[d][0]-size*mean_g[0]*mean_in[d])+minv[0][1]*(sum[d][0]-size*mean_g[0]*mean_in[d])+minv[0][2]*(sum[d][0]-size*mean_g[0]*mean_in[d]);
			a_temp[d][1]=minv[1][0]*(sum[d][1]-size*mean_g[1]*mean_in[d])+minv[1][1]*(sum[d][1]-size*mean_g[1]*mean_in[d])+minv[1][2]*(sum[d][1]-size*mean_g[1]*mean_in[d]);
			a_temp[d][2]=minv[2][0]*(sum[d][2]-size*mean_g[2]*mean_in[d])+minv[2][1]*(sum[d][2]-size*mean_g[2]*mean_in[d])+minv[2][2]*(sum[d][2]-size*mean_g[2]*mean_in[d]);

			LOG(INFO)<<a_temp[0][0];
			LOG(INFO)<<a_temp[0][1];
			LOG(INFO)<<a_temp[0][2];

			a_temp[d][0]/=size;
			a_temp[d][1]/=size;
			a_temp[d][2]/=size;

		}

		float au = 0;

		for (int d = 0; d < bpp; ++d) {
			au = 0;
			au += a_temp[d][0]*mean_g[0]+a_temp[d][1]*mean_g[1]+a_temp[d][2]*mean_g[2];
			b_temp[d] = mean_in[d] - au; 
		}


		for (int d = 0; d < bpp; ++d) {
			

			a_r[y*line_stride+x*bpp+d] = a_temp[d][0];
			a_g[y*line_stride+x*bpp+d] = a_temp[d][1];
			a_b[y*line_stride+x*bpp+d] = a_temp[d][2];

			b[y*line_stride+x*bpp+d] = b_temp[d];

		}
	
	




















































/*
//	unique_ptr<float[]> a_r(new float[w*h*bpp]);
//	unique_ptr<float[]> a_g(new float[w*h*bpp]);
//	unique_ptr<float[]> a_b(new float[w*h*bpp]);
//	unique_ptr<float[]> b(new float[w*h*bpp]);
	device_manager->ReadMemory(a_r.get(), *d_a_r.get(), w*h*bpp*sizeof(float));
	device_manager->ReadMemory(a_g.get(), *d_a_g.get(), w*h*bpp*sizeof(float));
	device_manager->ReadMemory(a_b.get(), *d_a_b.get(), w*h*bpp*sizeof(float));
	device_manager->ReadMemory(b.get(), *d_b.get(), w*h*bpp*sizeof(float));
	for (int y = offset; y < h-offset; ++y) {
		for (int x = offset; x < w-offset; ++x) {
			for (int d = 0; d < bpp; ++d) {
				LOG(INFO)<<"a_r["<<y<<"]["<<x<<"]["<<d<<"]="<<a_r[y*line_stride+x*bpp+d];
				LOG(INFO)<<"a_g["<<y<<"]["<<x<<"]["<<d<<"]="<<a_g[y*line_stride+x*bpp+d];
				LOG(INFO)<<"a_b["<<y<<"]["<<x<<"]["<<d<<"]="<<a_b[y*line_stride+x*bpp+d];

				LOG(INFO)<<"b["<<y<<"]["<<x<<"]["<<d<<"]="<<b[y*line_stride+x*bpp+d];
			}
		}
	}
*/
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
