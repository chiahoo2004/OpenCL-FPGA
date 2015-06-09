__kernel void guidedtwo(
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
	int d = get_global_id(2);

	int length = offset*2+1;
	int size = length * length;
	
	if (x < work_w && y < work_h && d < 3) {
		x += offset;
		y += offset;

		int pixel_output = 0;

		float sum_a = 0;
		float sum_b = 0;
		float mean_a, mean_b;

		for (int i = -offset; i <= offset; i++) {
			for(int j = -offset; j <= offset; j++) {
				sum_a += a[(y+j)*line_stride+(x+i)*bpp+d];
				sum_b += b[(y+j)*line_stride+(x+i)*bpp+d];

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
		out[y*line_stride+x*bpp+d] = (pixel_output&0xffffff00)? ~(pixel_output>>24): pixel_output;
	}

//	out[y*line_stride+x*bpp+d] = in[y*line_stride+x*bpp+d];
}








