__kernel void guided(
		__global const float *in,
		__global float *out,
		const int offset,
		const float epsilon,
		const int work_w,
		const int work_h,
		const int bpp,
		const int line_stride,
		__global float *a,
		__global float *b,
		__global float *I
	) {

	int x = get_global_id(0);
	int y = get_global_id(1);

	int length = offset*2+1;
	int size = length * length;

	if (x < work_w && y < work_h) {
		x += offset;
		y += offset;

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
					sum_in += in[(y+a)*line_stride+(x*bpp+d+b*bpp)]; 
					
					sum += I[(y+a)*line_stride+(x*bpp+d+b*bpp)] * in[(y+a)*line_stride+(x*bpp+d+b*bpp)];

					#ifdef DEBUG1
					DLOG(INFO)<<"I["<<(y+a)*line_stride+(x*bpp+d+b*bpp)<<"] = "<<I[(y+a)*line_stride+(x*bpp+d+b*bpp)]<<endl;
					DLOG(INFO)<<"image["<<(y+a)*line_stride+(x*bpp+d+b*bpp)<<"] = "<<in[(y+a)*line_stride+(x*bpp+d+b*bpp)]<<endl;
					#endif

				}
			}

			mean_g = sum_g/size;
			squaremean_g = square_g/size;
			mean_in = sum_in/size;
			variance_g = squaremean_g - mean_g * mean_g;
			

			a_temp = (sum - size*mean_g*mean_in) / (size*(variance_g+epsilon));
			b_temp = mean_in - a_temp*mean_g;
			
			a[y*line_stride+x*bpp+d] = a_temp;
			b[y*line_stride+x*bpp+d] = b_temp;
		}
	}

}


