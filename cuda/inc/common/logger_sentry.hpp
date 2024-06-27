#ifndef _LOGGER_SENTRY_H
#define _LOGGER_SENTRY_H

#include "logger.hpp"

void ssc_log_start_sentry(const char* project, const char* version, const char* level, const char* telem_key);
void ssc_log_stop_sentry();

void ssc_event_start_sentry(const char *evt_name, const std::vector<ssc_param_t> &params);
void ssc_event_stop_sentry();

#endif //_LOGGER_SENTRY_H
