#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#include <time.h>
#include <stdio.h>

double timestamp;

#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year   = wtm.wYear - 1900;
    tm.tm_mon   = wtm.wMonth - 1;
    tm.tm_mday   = wtm.wDay;
    tm.tm_hour   = wtm.wHour;
    tm.tm_min   = wtm.wMinute;
    tm.tm_sec   = wtm.wSecond;
    tm. tm_isdst  = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

double get_time()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

void start_time()
{

    timestamp=get_time();
}

void end_time(const char* str)
{
    timestamp=get_time()-timestamp;
    printf("%s TIME COST: %.6lfs\n",str,timestamp);
}