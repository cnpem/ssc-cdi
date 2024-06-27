#ifndef _LOGGER_H
#define _LOGGER_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <spdlog/formatter.h> // might be replaced by fmt std::format c++20 soon, or fmt library if needed

using std::string;
using fmt::format;

extern "C" {
void ssc_log_start(const char* level);
void ssc_log_stop();
}

void ssc_warning(const string& msg);
void ssc_error(const string& msg);
void ssc_info(const string& msg);
void ssc_debug(const string& msg);

void _ssc_assert(bool assertion,
        const std::string& log_msg = "",
        const char *file = __FILE__,
        const int line = __LINE__);
void _ssc_cufft_check(cufftResult_t fftres,
        const char *file = __FILE__,
        const int line = __LINE__);
void _ssc_cuda_check(cudaError_t cudares,
        const char *file = __FILE__,
        const int line = __LINE__);


#define ssc_assert(assertion, log_msg) _ssc_assert(assertion, log_msg, __FILE__, __LINE__)
#define ssc_cuda_check(res) _ssc_cuda_check(res, __FILE__, __LINE__)
#define ssc_cufft_check(res) _ssc_cufft_check(res, __FILE__, __LINE__)

#endif //_LOGGER_H
