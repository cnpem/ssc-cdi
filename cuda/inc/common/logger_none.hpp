#ifndef _LOGGER_SENTRY_H
#define _LOGGER_SENTRY_H

#include "logger.hpp"

inline void ssc_log_start_none(const char* project, const char* version, const char* level, const char* telem_key) {}
inline void ssc_log_stop_none() {}

inline void ssc_event_start_none(const char *evt_name, const std::vector<ssc_param_t> &params) {}
inline void ssc_event_stop_none() {}

#endif //_LOGGER_SENTRY_H
