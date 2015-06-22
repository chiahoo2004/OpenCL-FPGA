#pragma once
#include "filter.h"
#include <memory>
using std::unique_ptr;

struct GuidedFilter: public Filter {
	struct Parameter {
		float epsilon;
		int radius;
	};
	virtual void Run_cxx(const float *image_in, float* image_out);
	virtual void Run_ocl(const float *image_in, float* image_out);
	virtual void SetDimension(const int w, const int h, const int channel);
	void SetParameter(const Parameter &param) {
		param_ = param;
	}
private:
	unique_ptr<float[]> a, b;
	Parameter param_;
};
