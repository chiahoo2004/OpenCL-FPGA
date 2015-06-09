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

	vector<vector<vector<float> > > a;
	vector<vector<vector<float> > > b;
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
/*
	const int image_size = w*h*bpp;
	unique_ptr<float**[]> a(new float**[w]);
	unique_ptr<float**[]> b(new float**[w]);
	unique_ptr<float*[]> buffer(new float*[w*h]);
	unique_ptr<float[]> buffer1(new float[w*h*bpp]);
	for (int i = 0; i < w; ++i) {
		a[i] = buffer.get() + h*i;
		b[i] = buffer.get() + h*i;
		for(int j=0; j<h; ++j) {
			a[i][j] =  buffer1.get() + h*i + bpp*j;
			b[i][j] =  buffer1.get() + h*i + bpp*j;
		}
	}
*/
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
			
				a[x][y][d] = a_temp;
				b[x][y][d] = b_temp;
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
						sum_a += a[x+i][y+j][d];
						sum_b += b[x+i][y+j][d];
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

	auto d_a = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*bpp*sizeof(float));
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
	const size_t block_dim[3] = {32, 16, 1};
	const size_t grid_dim[3] = {CeilDiv(w, block_dim[0])*block_dim[0], CeilDiv(h, block_dim[1])*block_dim[1], bpp};

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
			{d_a.get(), sizeof(cl_mem)},
			{d_b.get(), sizeof(cl_mem)},
			{d_I.get(), sizeof(cl_mem)}
		},
		3, grid_dim, nullptr, block_dim
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
			{d_a.get(), sizeof(cl_mem)},
			{d_b.get(), sizeof(cl_mem)},
			{d_I.get(), sizeof(cl_mem)}
		},
		3, grid_dim, nullptr, block_dim
	);

	device_manager->ReadMemory(image_out, *d_out.get(), w*h*bpp*sizeof(float));
//	device_manager->ReadMemory(image_out, *d_in.get(), w*h*bpp*sizeof(float));
}
