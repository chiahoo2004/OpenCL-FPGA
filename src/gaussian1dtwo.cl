__kernel void gaussian1dtwo(
		__global const float *in,
		__global float *out,
		const int radius,
		const int work_w,
		const int work_h,
		const int bpp,
		const int line_stride,
		__constant float *range_gaussian_table,
		__global float *mid
	) {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int d = get_global_id(2);

	if (x < work_w && y < work_h && d < 3) {
		x += radius;
		y += radius;
		
		float weight_sum = 0.0f;
		float weight_pixel_sum = 0.0f;
		for (int i = -radius; i <= radius; ++i) {
			int range_diff = abs(i);
			weight_sum += range_gaussian_table[range_diff];
			weight_pixel_sum += range_gaussian_table[range_diff] * in[y*line_stride+((x+i)*bpp)+d];
		}

		const int pixel_output = weight_pixel_sum/weight_sum + 0.5f;
		out[y*line_stride+x*bpp+d] = ((int)pixel_output&0xffffff00)? ~((int)pixel_output>>24): (int)pixel_output;

	}
}


