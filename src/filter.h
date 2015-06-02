#pragma once
class Filter {
protected:
	int w_, h_, channel_;
public:
	virtual void SetDimension(const int w, const int h, const int channel) = 0;
	virtual void Run_cxx(const float *image_in, float *image_out) = 0;
	virtual void Run_ocl(const float *image_in, float *image_out) = 0;
	Filter(): w_(0), h_(0), channel_(0) {}
};
