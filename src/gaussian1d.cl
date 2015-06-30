__kernel void gaussian1d(
		__global const float *in,
		__global float *out,
		const int radius,
		const int work_w,
		const int work_h,
		const int bpp,
		const int line_stride,
		__constant float *range_gaussian_table
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < work_w && y < work_h) {
		x += radius;
		y += radius;

		int w = work_w + 2*radius;
		int h = work_h + 2*radius;

		for (int d=0;d<bpp;d++) {
			float weight_sum = 0.0f;
			float weight_pixel_sum = 0.0f;
			
			for (int i = -radius; i <= radius; ++i) {
				int range_diff = abs(i);
				weight_sum += range_gaussian_table[range_diff];
				weight_pixel_sum += range_gaussian_table[range_diff] * in[(y)*w+(x+i)+d*w*h];
			}

			const int mid_output = weight_pixel_sum/weight_sum + 0.5f;
			out[x*w+y+d*w*h] = ((int)mid_output&0xffffff00)? ~((int)mid_output>>24): (int)mid_output;
		}
	}
}

