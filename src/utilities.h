#pragma once
#include <cmath>
#include <algorithm>
using namespace std;

template <class Int=int> Int ClampToUint8(Int x)
{
	const Int mask = 0xff;
	return (x&~mask)? ((~x)>>(sizeof(Int)*8-1) & mask): x;
}

void FillBoundary(unsigned char *image_out, int offset, int w, int h, int bpp);
