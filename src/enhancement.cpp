#include "utilities.h"
#include "enhancement.h" 
#include <glog/logging.h>

//#define DEBUG1 1

void Enhance(unsigned char **images_clear_to_blur, unsigned char *out, const float *weights, const int w, const int h, const int bpp, const int layer)
{
	vector<vector<float> > color_img_diff;
	color_img_diff.resize(layer-1);
	for (int i = 0; i < layer-1; ++i)
		color_img_diff[i].resize(w*h*bpp); 
	for (int i = w*bpp; i < w*(h-1)*bpp; ++i) {
		// TODO!
		float accumulate = 0.0f;
		for (int j = 0; j < layer-1; ++j){
			color_img_diff[j][i] = images_clear_to_blur[j][i] - images_clear_to_blur[j+1][i];

			#ifdef DEBUG1
				LOG(INFO)<<"color_img_diff["<<j<<"]["<<i<<"] ("<<color_img_diff[j][i]<<") = images_clear_to_blur["<<j<<"]["<<i<<"] ("<<(double)images_clear_to_blur[j][i]<<")"
				<<" - images_clear_to_blur["<<j+1<<"]["<<i<<"] ("<<(double)images_clear_to_blur[j+1][i]<<")"<<endl;
			#endif

		}
		accumulate += weights[0]*images_clear_to_blur[0][i];

		#ifdef DEBUG1
			DLOG(INFO)<<"accumulate += weights[0] ("<<weights[0]<<") * images_clear_to_blur[0]["<<i<<"] ("<<(double)images_clear_to_blur[0][i]<<")"<<endl;
		#endif

		for (int j = 0; j < layer-1; ++j){
			accumulate += weights[j+1] * color_img_diff[j][i];
			#ifdef DEBUG1
				DLOG(INFO)<<"accumulate += weights["<<j+1<<"] ("<<weights[j+1]<<") * color_img_diff["<<j<<"]["<<i<<"] ("<<color_img_diff[j][i]<<")"<<endl;
			#endif
		}
		out[i] = ClampToUint8((int)accumulate);

		#ifdef DEBUG1
			DLOG(INFO)<<"out["<<i<<"] = "<<(double)out[i]<<endl;
		#endif
	}
}
