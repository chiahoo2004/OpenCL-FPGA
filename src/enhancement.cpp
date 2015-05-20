#include "utilities.h"
#include "enhancement.h" 
#include "filter.h"
#include <glog/logging.h>
#include <memory>

void Enhance(
	const float *img_in, float *img_out, const float *weights,
	const int w, const int h, const int channel, const int n_diff_layer, Filter *filter
)
{
	const int image_size = w*h*channel;
	unique_ptr<float[]> buffer(new float[image_size*n_diff_layer]);
	unique_ptr<float*[]> layers(new float*[n_diff_layer+1]); // blur to clear
	// const_cast is OK because we won't obey it
	layers[n_diff_layer] = const_cast<float*>(img_in);
	for (int i = n_diff_layer-1; i >= 0; --i) {
		layers[i] = buffer.get() + image_size*i;
		filter->Run(layers[i+1], layers[i]);
	}
	for (int i = 0; i < image_size; ++i) {
		float accumulate = layers[0][i];
		for (int j = 0; j < n_diff_layer; ++j){
			accumulate += weights[j] * (layers[j+1][i]-layers[j][i]);
		}
		img_out[i] = accumulate;
	}
}
