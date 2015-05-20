#pragma once
#include <chrono>
typedef std::chrono::time_point<std::chrono::high_resolution_clock> Clock;

static inline Clock GetNow()
{
	return std::chrono::high_resolution_clock::now();
}

static inline long long DiffUsInLongLong(const Clock tic, const Clock toc)
{
	return std::chrono::duration_cast<std::chrono::microseconds>(toc-tic).count();
}
