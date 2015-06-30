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

		int w = work_w + 2*offset;
		int h = work_h + 2*offset;

		for (int d = 0; d < bpp; ++d) {
			int pixel_output = 0;

			float sum_a[3]={};
			float sum_b = 0;
			float mean_a[3]={};
			float mean_b = 0;

			for (int i = -offset; i <= offset; i++) {
				for(int j = -offset; j <= offset; j++) {
					sum_a[0] += a_r[(y+j)*w+(x+i)+d*w*h];
					sum_a[1] += a_g[(y+j)*w+(x+i)+d*w*h];
					sum_a[2] += a_b[(y+j)*w+(x+i)+d*w*h];
					sum_b += b[(y+j)*w+(x+i)+d*w*h];
				}
			}	

			mean_a[0] = sum_a[0] / size;
			mean_a[1] = sum_a[1] / size;
			mean_a[2] = sum_a[2] / size;
			mean_b = sum_b / size;

			pixel_output = mean_a[0]*I[y*w+x+ 0*w*h]+mean_a[1]*I[y*w+x+ 1*w*h]+mean_a[2]*I[y*w+x+ 2*w*h]+mean_b;
			out[y*w+x+d*w*h] = (pixel_output&0xffffff00)? ~(pixel_output>>24): pixel_output;
		}
	}

}
