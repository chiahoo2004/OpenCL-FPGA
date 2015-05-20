#pragma once
#include "filter.h"

struct GaussianFilter: public Filter {
	struct Parameter {
		float r_sigma;
		int radius;
	};
	virtual void Run(const float *image_in, float* image_out);
	virtual void SetDimension(const int w, const int h, const int channel);
	void SetParameter(const Parameter &param) {
		param_ = param;
	}
private:
	Parameter param_;
};

