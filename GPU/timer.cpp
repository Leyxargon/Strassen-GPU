#include "timer.hpp"

Timer::Timer() {
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
}

Timer::~Timer() {
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
}

void Timer::start() {
	cudaEventRecord(begin, 0);
}

void Timer::stop() {
	cudaEventRecord(end, 0);
}

float Timer::get() {
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, begin, end);
	return time;
}