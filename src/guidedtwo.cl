__kernel void guidedtwo(
		__global const float *in,
		__global float *out,
		const int offset,
		const float epsilon,
		const int work_w,
		const int work_h,
		const int bpp,
		const int line_stride,
		__global float *a_r,
		__global float *a_g,
		__global float *a_b,
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
			int pixel_output = 0;

			float sum_a[3]={};
			float sum_b = 0;
			float mean_a[3]={};
			float mean_b = 0;

			for (int i = -offset; i <= offset; i++) {
				for(int j = -offset; j <= offset; j++) {
					sum_a[0] += a_r[(y+j)*line_stride+(x+i)*bpp+d];
					sum_a[1] += a_g[(y+j)*line_stride+(x+i)*bpp+d];
					sum_a[2] += a_b[(y+j)*line_stride+(x+i)*bpp+d];
					sum_b += b[(y+j)*line_stride+(x+i)*bpp+d];

					#ifdef DEBUG2
					DLOG(INFO)<<"sum_a += "<<sum_a<<endl;
					DLOG(INFO)<<"sum_b += "<<sum_b<<endl;
					#endif
				}
			}	

			mean_a[0] = sum_a[0] / size;
			mean_a[1] = sum_a[1] / size;
			mean_a[2] = sum_a[2] / size;
			mean_b = sum_b / size;
			#ifdef DEBUG2
			DLOG(INFO)<<"mean_a["<<y*line_stride+x*bpp+d<<"] = "<<mean_a<<endl;
			DLOG(INFO)<<"mean_b["<<y*line_stride+x*bpp+d<<"] = "<<mean_b<<endl;
			#endif

			pixel_output = mean_a[0]*I[y*line_stride+x*bpp+0]+mean_a[1]*I[y*line_stride+x*bpp+1]+mean_a[2]*I[y*line_stride+x*bpp+2]+mean_b;
			out[y*line_stride+x*bpp+d] = (pixel_output&0xffffff00)? ~(pixel_output>>24): pixel_output;
		}
	}

//	out[y*line_stride+x*bpp+d] = in[y*line_stride+x*bpp+d];
}








