#ifndef _UTILS_H
#define _UTILS_H

#include <chrono>

using ssc_timepoint = std::chrono::time_point<std::chrono::system_clock>;

inline ssc_timepoint sscTime() {
    return std::chrono::system_clock::now();
}

inline float ssc_diff_time(ssc_timepoint t0, ssc_timepoint t1) {
    std::chrono::duration<float> diff(t1 - t0);
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}


#endif // _UTILS_H
