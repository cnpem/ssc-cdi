#include "logger.hpp"

#include <pwd.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <mutex>
#include <unistd.h>
#include <csignal>
#include <string>
#include <atomic>

std::mutex evt_mutex;
std::atomic_bool log_active;

void ssc_log_start(const char* level) {
    if (log_active) {
        sscWarning("Log already started. Ignoring start.");
        return;
    }

    // logger
    spdlog::set_pattern("%[%H:%M:%S:%f] %^%g:%! [%l] [thread %t]%$ %v");
    spdlog::set_level(spdlog::level::from_str(level));

    log_active = true;
}

void ssc_log_stop() {
    if (!log_active) {
        sscWarning("Log not started. Ignoring stop.");
    }
    log_active = false;
    spdlog::drop_all();
    spdlog::shutdown();
}

void sscWarning(const string& msg) {
    if (log_active)
        spdlog::warn(msg);
}

void sscError(const string& msg) {
    if (log_active)
        spdlog::error(msg);
}

void sscInfo(const string& msg) {
    if (log_active)
        spdlog::info(msg);
}

void sscDebug(const string& msg) {
    if (log_active)
        spdlog::debug(msg);
}

const char* cufftGetErrorString(cufftResult s) {
    switch (s) {
        case CUFFT_SUCCESS:
            return "Success";
        case CUFFT_INVALID_PLAN:
            return "Invalid plan handle";
        case CUFFT_ALLOC_FAILED:
            return "Alloc failed";
        case CUFFT_INVALID_TYPE:
            return "Invalid type";
        case CUFFT_INVALID_VALUE:
            return "Invalid value (bad pointer)";
        case CUFFT_INTERNAL_ERROR:
            return "Internal driver error";
        case CUFFT_EXEC_FAILED:
            return "Failed to execute an FFT";
        case CUFFT_SETUP_FAILED:
            return "Failed to initialize";
        case CUFFT_INVALID_SIZE:
            return "Invalid FFT size";
        default:
            return "Unknown error (may God have mercy on your soul)";
    }
}

void _sscAssert(bool assertion,
        const std::string& log_msg,
        const char *file, const int line) {
        if (!assertion) {
            sscError(format("{} ({}): *** assertion error: {}",
                        file, line, log_msg.c_str()));
            raise(SIGABRT);
        }
}

void _sscCufftCheck(cufftResult fftres,
        const char *file, const int line) {
    if (fftres != CUFFT_SUCCESS) {
        sscError(format("{} ({}) => *** cufftError: {}",
                    file, line, cufftGetErrorString(fftres)));
        raise(SIGABRT);
    }
}

void _sscCudaCheck(cudaError_t cudares,
        const char *file, const int line) {
    if (cudares != cudaSuccess) {
        int device;
        sscError(format("{} ({}) => *** cudaError: {}",
                    file, line, cudaGetErrorString(cudares)));
        raise(SIGABRT);
    }
}
