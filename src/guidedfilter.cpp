#include "guidedfilter.h"
#include "utilities.h"
#include <glog/logging.h>
#include <cmath>
#include <memory>
using namespace std;

//#define DEBUG1 1  
//#define DEBUG2 1  

void GuidedFilter::Run(unsigned char *image_in, unsigned char* image_out, unsigned char* I, vector<vector<vector<double> > >& a, vector<vector<vector<double> > >& b, int windowSize, double epsilon, int w, int h, int bpp)
{
	int size = windowSize * windowSize;
	int offset = (windowSize-1)/2;

	const int line_stride = bpp*w;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = bpp*offset; x < line_stride-bpp*offset; ++x) {

			double sum_g = 0;
			double square_g = 0;
			double sum_in = 0;
			double sum = 0;
			
			double mean_g, squaremean_g, mean_in, variance_g, a_temp, b_temp;

			for (int a = -offset; a <= offset; a++) {
				for(int b = -offset; b <= offset; b++) {

					sum_g += (double)I[(y+a)*line_stride+(x+b*bpp)];
					
					square_g += (double)I[(y+a)*line_stride+(x+b*bpp)] * (double)I[(y+a)*line_stride+(x+b*bpp)];
					sum_in += (double)image_in[(y+a)*line_stride+(x+b*bpp)]; 
					
					sum += (double)I[(y+a)*line_stride+(x+b*bpp)] * (double)image_in[(y+a)*line_stride+(x+b*bpp)];

					#ifdef DEBUG1
					DLOG(INFO)<<"I["<<(y+a)*line_stride+(x+b*bpp)<<"] = "<<(double)I[(y+a)*line_stride+(x+b*bpp)]<<endl;
					DLOG(INFO)<<"image["<<(y+a)*line_stride+(x+b*bpp)<<"] = "<<(double)image_in[(y+a)*line_stride+(x+b*bpp)]<<endl;
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
		


			switch(x%bpp)
			{
				case 0:
					a[x/bpp][y][0] = a_temp;
					b[x/bpp][y][0] = b_temp;
					#ifdef DEBUG1
					DLOG(INFO)<<"a["<<x/bpp<<"]["<<y<<"]["<<x%bpp<<"] = "<<a_temp<<endl;
					DLOG(INFO)<<"b["<<x/bpp<<"]["<<y<<"]["<<x%bpp<<"] = "<<b_temp<<endl;
					#endif
					break;
				case 1:
					a[x/bpp][y][1] = a_temp;
					b[x/bpp][y][1] = b_temp;
					#ifdef DEBUG1
					DLOG(INFO)<<"a["<<x/bpp<<"]["<<y<<"]["<<x%bpp<<"] = "<<a_temp<<endl;
					DLOG(INFO)<<"b["<<x/bpp<<"]["<<y<<"]["<<x%bpp<<"] = "<<b_temp<<endl;
					#endif
					break;
				case 2:
					a[x/bpp][y][2] = a_temp;
					b[x/bpp][y][2] = b_temp;
					#ifdef DEBUG1
					DLOG(INFO)<<"a["<<x/bpp<<"]["<<y<<"]["<<x%bpp<<"] = "<<a_temp<<endl;
					DLOG(INFO)<<"b["<<x/bpp<<"]["<<y<<"]["<<x%bpp<<"] = "<<b_temp<<endl;
					#endif
					break;
				default:
					DLOG(INFO)<<"error"<<endl;
					break;
			}

			#ifdef DEBUG1
			DLOG(INFO)<<"pause"<<endl;
			fgetc(stdin);
			#endif

		}
	}

	int pixel_output = 0;
	for (int y = offset; y < h-offset; ++y) {
		for (int x = bpp*offset; x < line_stride-bpp*offset; ++x) {

			double sum_a = 0;
			double sum_b = 0;
			double mean_a, mean_b;

			for (int i = -offset; i <= offset; i++) {
			                for(int j = -offset; j <= offset; j++) {
			                	sum_a += a[(x/bpp)+i][y+j][x%bpp];
			                	sum_b += b[(x/bpp)+i][y+j][x%bpp];
			                	#ifdef DEBUG2
			                	DLOG(INFO)<<"sum_a += "<<sum_a<<endl;
			                	DLOG(INFO)<<"sum_b += "<<sum_b<<endl;
			                	#endif
				}
			}	

			mean_a = sum_a / size;
			mean_b = sum_b / size;
			#ifdef DEBUG2
	                	DLOG(INFO)<<"mean_a["<<y*line_stride+x*bpp<<"] = "<<mean_a<<endl;
	                	DLOG(INFO)<<"mean_b["<<y*line_stride+x*bpp<<"] = "<<mean_b<<endl;
	                	#endif

			pixel_output = mean_a*(double)I[y*line_stride+x] +mean_b;
			image_out[y*line_stride+x] = ClampToUint8(abs(pixel_output)*1);
			#ifdef DEBUG2
	                	DLOG(INFO)<<"pixel_output["<<y*line_stride+x<<"] ("<<(double)image_out[y*line_stride+x]<<") = mean_a ("<<mean_a<<")   *   I["<<y*line_stride+x<<"] ("<<(double)I[y*line_stride+x]<<")   +   mean_b ("<<mean_b<<")"<<endl;
	                	DLOG(INFO)<<"pause"<<endl;
	                	fgetc(stdin);
	                	#endif
	                	
		}
	}
}
