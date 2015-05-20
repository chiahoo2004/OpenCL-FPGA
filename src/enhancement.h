#pragma once
class Filter;
void Enhance(
	const float *img_in, float *img_out, const float *weights,
	const int w, const int h, const int channel, const int n_diff_layer, Filter *filter
);
