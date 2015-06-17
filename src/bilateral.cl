__kernel void bilateral(
		__global const float *in,
		__global float *out,
		const int r,
		const int work_w,
		const int work_h,
		const int bpp,
		const int row_stride,
		const float color_sigma,
		__constant float *range_gaussian_table,
		__constant float *color_gaussian_table
) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < work_w && y < work_h) {
		x += r;
		y += r;

		int color_diff = 0;
		int color_weight = 0;

		for(int rgb=0;rgb<bpp;++rgb) {

			__global const float *base_in = &in[row_stride*bpp*y+bpp*x+rgb];
			__global float *base_out = &out[row_stride*bpp*y+bpp*x+rgb];
			float weight_sum = 0.0f;
			float weight_pixel_sum = 0.0f;

			for (int dy = -r; dy <= r; dy++) {
				for (int dx = -r; dx <= r; dx++) {
					int range_xdiff = abs(dx);
					int range_ydiff = abs(dy);
					

					if(rgb==0){

						for (int d = 0; d < bpp; ++d)
						{
							int diff = in[d+(x+dx)*bpp+(y+dy)*row_stride*bpp]-in[d+(x+dx)*bpp+(y+dy)*row_stride*bpp];
							int diff_1d = abs( diff );
							color_diff += diff_1d * diff_1d;
						}

						const float denominator_inverse = -1.0f / (2.0f * color_sigma * color_sigma);
						color_weight = exp(color_diff*denominator_inverse);

					}
					


					float weight =
						color_weight
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

}






