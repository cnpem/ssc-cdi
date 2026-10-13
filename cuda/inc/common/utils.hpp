#ifndef _UTILS_H
#define _UTILS_H

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda_runtime.h>

using ssc_timepoint = std::chrono::time_point<std::chrono::system_clock>;

inline ssc_timepoint sscTime() {
    return std::chrono::system_clock::now();
}

inline float sscDiffTime(ssc_timepoint t0, ssc_timepoint t1) {
    std::chrono::duration<float> diff(t1 - t0);
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

inline size_t sscSingleGpuAvailableMem() {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

inline size_t sscMultiGpuAvailableMem(const std::vector<int>& gpu_indices) {
    size_t free_mem = 0;
    for (int gpu_idx : gpu_indices) {
	cudaSetDevice(gpu_idx);
	free_mem += sscSingleGpuAvailableMem();
    }
    return free_mem;
}

inline size_t sscGpuTotalMem() {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}


#endif // _UTILS_H
