#ifndef __TIMER_H__
#define __TIMER_H__

#include <cuda_runtime.h>

class Timer {
private:
	cudaEvent_t begin;
	cudaEvent_t end;
	float time;
public:
	Timer();
	~Timer();
	void start();
	void stop();
	float get();
};

#include "timer.cpp"

#endif