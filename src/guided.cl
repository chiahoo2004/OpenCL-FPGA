__kernel void guided(
		__global const float *in,
		__global float *out,
		const int offset,
		const float epsilon,
		const int work_w,
		const int work_h,
		const int bpp,
		const int line_stride,
		__global float *a_r,
		__global float *a_g,
		__global float *a_b,
		__global float *b,
		__global float *I
	) {

	int x = get_global_id(0);
	int y = get_global_id(1);

	int length = offset*2+1;
	int size = length * length;

	if (x < work_w && y < work_h) {
		x += offset;
		y += offset;

		int w = work_w + 2*offset;
		int h = work_h + 2*offset;

		float sum_g[3]={};
		float square_g[3]={};
		float sum_in[3]={};
		float sum[3][3]={{0,0,0},{0,0,0},{0,0,0}};

		float corr_rg = 0;
		float corr_rb = 0;
		float corr_gb = 0;

		float mean_g[3]={};
		float squaremean_g[3]={};
		float mean_in[3]={};

		float var_I_rr = 0;
		float var_I_rg = 0;
		float var_I_rb = 0;
		float var_I_gg = 0;
		float var_I_gb = 0;
		float var_I_bb = 0;

		float a_temp[3][3]={}; 
		float b_temp[3]={};


		for (int d = 0; d < bpp; ++d) {

			for (int a = -offset; a <= offset; a++) {
				for(int b = -offset; b <= offset; b++) {

					sum_g[d] += I[(y+a)*w+(x+b)+d*w*h];
					
					square_g[d] += I[(y+a)*w+(x+b)+d*w*h] * I[(y+a)*w+(x+b)+d*w*h];
					sum_in[d] += in[(y+a)*w+(x+b)+d*w*h]; 
					
					sum[d][0] += I[(y+a)*w+(x+b)+ 0*w*h] * in[(y+a)*w+(x+b)+d*w*h];
					sum[d][1] += I[(y+a)*w+(x+b)+ 1*w*h] * in[(y+a)*w+(x+b)+d*w*h];
					sum[d][2] += I[(y+a)*w+(x+b)+ 2*w*h] * in[(y+a)*w+(x+b)+d*w*h];

					if (d==0) {
						corr_rg += I[(y+a)*w+(x+b)+ 0*w*h] * I[(y+a)*w+(x+b)+ 1*w*h];
						corr_rb += I[(y+a)*w+(x+b)+ 0*w*h] * I[(y+a)*w+(x+b)+ 2*w*h];
						corr_gb += I[(y+a)*w+(x+b)+ 1*w*h] * I[(y+a)*w+(x+b)+ 2*w*h];
					}

				}
			}

			mean_g[d] = sum_g[d]/size;
			squaremean_g[d] = square_g[d]/size;
			mean_in[d] = sum_in[d]/size;

		}

		corr_rg /= size; 
		corr_rb /= size; 
		corr_gb /= size; 

		var_I_rr = squaremean_g[0] - mean_g[0] * mean_g[0];
		var_I_rg = corr_rg - mean_g[0] * mean_g[1];
		var_I_rb = corr_rb - mean_g[0] * mean_g[2];
		var_I_gg = squaremean_g[1] - mean_g[1] * mean_g[1];
		var_I_gb = corr_gb - mean_g[1] * mean_g[2];
		var_I_bb = squaremean_g[2] - mean_g[2] * mean_g[2];


		float m[3][3];
		m[0][0] = var_I_rr + epsilon;
		m[0][1] = var_I_rg;
		m[0][2] = var_I_rb;
		m[1][0] = var_I_rg;
		m[1][1] = var_I_gg + epsilon;
		m[1][2] = var_I_gb;
		m[2][0] = var_I_rb;
		m[2][1] = var_I_gb;
		m[2][2] = var_I_bb + epsilon; 


		// computes the inverse of a matrix m
		float det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
		             m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
		             m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

		float invdet = 1 / det;

		float minv[3][3]; // inverse of matrix m
		minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
		minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
		minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
		minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
		minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
		minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
		minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
		minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
		minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;

		for (int d = 0; d < bpp; ++d) {


			a_temp[d][0]=minv[0][0]*(sum[d][0]-size*mean_g[0]*mean_in[d])+minv[0][1]*(sum[d][0]-size*mean_g[0]*mean_in[d])+minv[0][2]*(sum[d][0]-size*mean_g[0]*mean_in[d]);
			a_temp[d][1]=minv[1][0]*(sum[d][1]-size*mean_g[1]*mean_in[d])+minv[1][1]*(sum[d][1]-size*mean_g[1]*mean_in[d])+minv[1][2]*(sum[d][1]-size*mean_g[1]*mean_in[d]);
			a_temp[d][2]=minv[2][0]*(sum[d][2]-size*mean_g[2]*mean_in[d])+minv[2][1]*(sum[d][2]-size*mean_g[2]*mean_in[d])+minv[2][2]*(sum[d][2]-size*mean_g[2]*mean_in[d]);

			a_temp[d][0]/=size;
			a_temp[d][1]/=size;
			a_temp[d][2]/=size;

		}

		float au = 0;

		for (int d = 0; d < bpp; ++d) {
			au = 0;
			au += a_temp[d][0]*mean_g[0]+a_temp[d][1]*mean_g[1]+a_temp[d][2]*mean_g[2];
			b_temp[d] = mean_in[d] - au; 
		}


		for (int d = 0; d < bpp; ++d) {
			

			a_r[y*w+x+d*w*h] = a_temp[d][0];
			a_g[y*w+x+d*w*h] = a_temp[d][1];
			a_b[y*w+x+d*w*h] = a_temp[d][2];

			b[y*w+x+d*w*h] = b_temp[d];

		}
	
	}

}


