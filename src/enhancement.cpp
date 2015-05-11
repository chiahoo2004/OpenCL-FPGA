#include "utilities.h"
#include "enhancement.h"

void Enhance(const char **images_clear_to_blur, char *out, const float *weights, const int w, const int h, const int layer)
{
	for (int i = 0; i < w*h*bpp; ++i) {
		// TODO!
		float accumulate = 0.0f;
		out[i] = ClampToUint8((int)accumulate);
	}
}
