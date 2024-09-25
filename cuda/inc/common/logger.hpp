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

void sscWarning(const string& msg);
void sscError(const string& msg);
void sscInfo(const string& msg);
void sscDebug(const string& msg);

void _sscAssert(bool assertion,
        const std::string& log_msg = "",
        const char *file = __FILE__,
        const int line = __LINE__);
void _sscCufftCheck(cufftResult_t fftres,
        const char *file = __FILE__,
        const int line = __LINE__);
void _sscCudaCheck(cudaError_t cudares,
        const char *file = __FILE__,
        const int line = __LINE__);


#define sscAssert(assertion, log_msg) _sscAssert(assertion, log_msg, __FILE__, __LINE__)
#define sscCudaCheck(res) _sscCudaCheck(res, __FILE__, __LINE__)
#define sscCufftCheck(res) _sscCufftCheck(res, __FILE__, __LINE__)

#endif //_LOGGER_H
