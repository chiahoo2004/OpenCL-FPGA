__kernel void bilateral(
		__global const float *in,
		__global float *out,
		const int r,
		const int work_w,
		const int work_h,
		const int bpp,
		const int row_stride,
		__constant float *range_gaussian_table,
		__constant float *color_gaussian_table
) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int rgb = get_global_id(2);

	if (x < work_w && y < work_h && rgb < 3) {
		x += r;
		y += r;
		__global const float *base_in = &in[row_stride*bpp*y+bpp*x+rgb];
		__global float *base_out = &out[row_stride*bpp*y+bpp*x+rgb];
		float weight_sum = 0.0f;
		float weight_pixel_sum = 0.0f;
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				int range_xdiff = abs(dx);
				int range_ydiff = abs(dy);
				int diff = base_in[0] - base_in[dy*row_stride*bpp+dx*bpp];
				int color_diff = abs(diff);
				float weight =
					color_gaussian_table[color_diff]
					* range_gaussian_table[range_xdiff]
					* range_gaussian_table[range_ydiff];
				weight_sum += weight;
				weight_pixel_sum += weight * base_in[dy*row_stride*bpp+dx*bpp];
			}
		}
		const int output_pixel = weight_pixel_sum/weight_sum + 0.5f;
		base_out[0] = (output_pixel&0xffffff00)? ~(output_pixel>>24): output_pixel;
	}
}