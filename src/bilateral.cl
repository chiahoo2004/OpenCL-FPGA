__kernel void bilateral(
		__global const float *in,
		__global float *out,
		const int r,
		const int work_w,
		const int work_h,
		const int bpp,
		const int line_stride,
		const float color_sigma,
		__constant float *range_gaussian_table,
		__constant float *color_gaussian_table
) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < work_w && y < work_h) {
		x += r;
		y += r;

		float color_diff = 0;
		float color_weight = 0;
		float weight_sum = 0.0f;
		float weight_pixel_sum[3] = {};

		__global const float *base_in = &in[line_stride*y+bpp*x];
		__global float *base_out = &out[line_stride*y+bpp*x];
		

		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				int range_xdiff = abs(dx);
				int range_ydiff = abs(dy);
				for (int d = 0; d < bpp; ++d)
				{
					float diff = base_in[d+dx*bpp+dy*line_stride]-base_in[d];
					color_diff += diff * diff;
				}

				const float denominator_inverse = -1.0f / (2.0f * color_sigma * color_sigma);
				color_weight = exp(color_diff*denominator_inverse);
				float weight =
					color_weight
					* range_gaussian_table[range_xdiff]
					* range_gaussian_table[range_ydiff];

				weight_sum += weight;
				for (int d = 0; d < bpp; ++d) {
					weight_pixel_sum[d] += weight * base_in[d+dy*line_stride+dx*bpp];
				}
				
			}
		}

		for (int d = 0; d < bpp; ++d) {
			const int output_pixel = weight_pixel_sum[d] / weight_sum + 0.5f;
			base_out[d] = (output_pixel&0xffffff00)? ~(output_pixel>>24): output_pixel;
		}
	}
}
