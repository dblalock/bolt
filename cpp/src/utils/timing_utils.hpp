//
//  timing_utils.cpp
//  Dig
//
//  Created by Davis Blalock on 2016-3-28
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef _TIMING_UTILS_HPP
#define _TIMING_UTILS_HPP

#include <chrono>
#include <iostream>

#ifdef _WIN32
    #include <intrin.h> // for cycle counting
#endif

namespace {

// cycle counting adapted from http://stackoverflow.com/a/13772771
#ifdef _WIN32 //  Windows
    static inline uint64_t time_now_cycles() { return __rdtsc(); }
#else //  Linux/GCC
    static inline uint64_t time_now_cycles() {
        unsigned int lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((uint64_t)hi << 32) | static_cast<uint64_t>(lo);
    }
#endif

using cputime_t = std::chrono::high_resolution_clock::time_point;
//#define clock std::chrono::high_resolution_clock // because so much typing

static inline cputime_t timeNow() {
	return std::chrono::high_resolution_clock::now();
}

static inline int64_t durationUs(cputime_t t1, cputime_t t0) {
    auto diff = t1 >= t0 ? t1 - t0 : t0 - t1; // = abs(t1 - t0);
    return std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    // double diffMicros = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    // return std::abs(diffMicros);

    // std::chrono::duration<uint64_t, std::micro> ret = t1 - t0;
    // return ret.count();
    // return std::abs(diffMicros);
    // return diffMicros;
}

static inline double durationMs(cputime_t t1, cputime_t t0) {
     return durationUs(t1, t0) / 1000.0;
//	double diffMicros = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
//	return std::abs(diffMicros) / 1000.0;
}

class EasyTimer {
public:
    using TimeT = double;
    EasyTimer(TimeT& write_to, bool add=false, bool ms=true):
        _write_here(write_to), _tstart(timeNow()), _add(add), _ms(ms) {}
    ~EasyTimer() {
        TimeT duration = static_cast<TimeT>(durationUs(_tstart, timeNow()));
        if (_ms) {
            duration /= 1000.0;
        }
        if (_add) {
            _write_here += duration;
        } else {
            _write_here = duration;
        }
    }
private:
    TimeT& _write_here;
    cputime_t _tstart;
    bool _add;
    bool _ms;
};

// WARNING: for some reason, PrintTimer's dtor often gets called immediately
// after the ctor, and so the time it prints is 0 or just a few micros
class PrintTimer {
public:
    PrintTimer(const std::string& msg): _msg(msg), _tstart(timeNow()) {}
    ~PrintTimer() {
        // std::cout << _tstart << " to " << timeNow() << "\n"; // TODO rm
        // auto elapsed = durationUs(_tstart, timeNow());
        // auto elapsed = durationUs(timeNow(), _tstart);
        if (_msg.size()) {
            auto elapsed = durationMs(_tstart, timeNow());
            std::cout << _msg << ":\t" << elapsed << "\tms\n";

//            auto elapsedUs = durationUs(_tstart, timeNow());
//            std::cout << _msg << ":\t" << elapsedUs << "\tus\n";
        }
    }
private:
    std::string _msg;
    cputime_t _tstart;
};

} // anon namespace
#endif // _TIMING_UTILS_HPP
