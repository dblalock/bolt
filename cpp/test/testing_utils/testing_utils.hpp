//
//  testing_utils.hpp
//  Dig
//
//  Created by DB on 10/22/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TESTING_UTILS_HPP
#define __TESTING_UTILS_HPP

#include <iostream>
#include <random>
#include <stdint.h>



#ifdef BLAZE
    #include "test/external/catch.hpp"
    #include "src/utils/eigen_utils.hpp"
    #include "src/utils/memory.hpp"
    #include "src/utils/timing_utils.hpp" // for profiling
#else
    #include "catch.hpp"
    #include "eigen_utils.hpp"
    #include "memory.hpp"
    #include "timing_utils.hpp" // for profiling
#endif


template<class DistT>
double prevent_optimizing_away_dists(DistT* dists, int64_t N,
    bool verbose=false)
{
    // count how many dists are above a random threshold; let's see you
    // optimize this away, compiler
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(0, 255);
    uint8_t thresh = static_cast<uint8_t>(d(gen));

    volatile int64_t count = 0;
    for (int64_t n = 0; n < N; n++) { count += dists[n] > thresh; }
    volatile double frac = static_cast<double>(count) / N;
    if (verbose) {
        printf("(%d%%) ", static_cast<int>(frac * 100));
    }
    if (frac == d(gen)) {
        printf("wow, frac was exactly equal to a random real number...\n");
    }
    return frac;
}

static inline void print_dist_stats(const std::string& name, int64_t N,
    double t_ms)
{
    double thruput_mil = N / (1e3 * t_ms); // millions
    if (thruput_mil  < .001) {
        printf("%s: %.2f (%.1f/s)\n", name.c_str(), t_ms, thruput_mil * 1e6);
    } else if (thruput_mil < 1) {
        printf("%s: %.2f (%.2fK/s)\n", name.c_str(), t_ms, thruput_mil * 1e3);
    } else if (thruput_mil < 100) {
        printf("%s: %.2f (%.2fM/s)\n", name.c_str(), t_ms, thruput_mil);
    } else {
        printf("%s: %.2f (%.1fM/s)\n", name.c_str(), t_ms, thruput_mil);
    }
}


#define PROFILE_DIST_COMPUTATION(NAME, NTRIALS, DISTS_PTR, NUM_DISTS, EXPR)  \
    do {                                                                \
        double __t_min = std::numeric_limits<double>::max();            \
        for (int __i = 0; __i < NTRIALS; __i++) {                       \
            double __t = 0;                                             \
            {                                                           \
                EasyTimer _(__t);                                       \
                (EXPR);                                                 \
            }                                                           \
            prevent_optimizing_away_dists(DISTS_PTR, NUM_DISTS);        \
            __t_min = __t < __t_min ? __t : __t_min;                    \
        }                                                               \
        print_dist_stats(                                               \
            NAME " (best of " #NTRIALS ")",                             \
            NUM_DISTS, __t_min);                                        \
    } while (0);


#define REPEATED_PROFILE_DIST_COMPUTATION(                              \
    NREPS, NAME, NTRIALS, DISTS_PTR, NUM_DISTS, EXPR)                   \
    do {                                                                \
        std::cout << NAME << "(" << NREPS << "x" << NTRIALS << "):";    \
        for (int __rep = 0; __rep < NREPS; __rep++) {                   \
            double __t_min = std::numeric_limits<double>::max();        \
            for (int __i = 0; __i < NTRIALS; __i++) {                   \
                double __t = 0;                                         \
                {                                                       \
                    EasyTimer _(__t);                                   \
                    (EXPR);                                             \
                }                                                       \
                prevent_optimizing_away_dists(DISTS_PTR, NUM_DISTS);    \
                __t_min = __t < __t_min ? __t : __t_min;                \
            }                                                           \
            printf(", %7.4f, (%.4e/s)", __t_min,                        \
                static_cast<double>(NUM_DISTS * 1e3 / __t_min));        \
        }                                                               \
        printf("\n");                                                   \
    } while (0);


#define PROFILE_DIST_COMPUTATION_LOOP(                                  \
    NAME, NTRIALS, DISTS_PTR, NUM_DISTS, NUM_LOOP_ITERS, EXPR)          \
    do {                                                                \
        double __t_min = std::numeric_limits<double>::max();            \
        for (int __i = 0; __i < NTRIALS; __i++) {                       \
            double __t = 0;                                             \
            auto t0 = timeNow();                                        \
            for (int i = 0; i < NUM_LOOP_ITERS; i++) {                  \
                (EXPR);                                                 \
            }                                                           \
            __t = durationUs(t0, timeNow());                            \
            prevent_optimizing_away_dists(DISTS_PTR, NUM_DISTS);        \
            __t_min = __t < __t_min ? __t : __t_min;                    \
        }                                                               \
        print_dist_stats(                                               \
            NAME " (best of " #NTRIALS ")",                             \
            NUM_LOOP_ITERS, __t_min / 1000.0);                          \
    } while (0);

#define REPEATED_PROFILE_DIST_COMPUTATION_LOOP(                         \
    NREPS, NAME, NTRIALS, DISTS_PTR, NUM_DISTS, NUM_LOOP_ITERS, EXPR)   \
    do {                                                                \
        std::cout << NAME << " (" << NREPS << "x" << NTRIALS << "): ";  \
        for (int __rep = 0; __rep < NREPS; __rep++) {                   \
            double __t_min = std::numeric_limits<double>::max();        \
            for (int __i = 0; __i < NTRIALS; __i++) {                   \
                double __t = 0;                                         \
                auto t0 = timeNow();                                    \
                for (int i = 0; i < NUM_LOOP_ITERS; i++) {              \
                    (EXPR);                                             \
                }                                                       \
                __t = durationUs(t0, timeNow());                        \
                prevent_optimizing_away_dists(DISTS_PTR, NUM_DISTS);    \
                __t_min = __t < __t_min ? __t : __t_min;                \
            }                                                           \
            printf("%.5g, (%lld/s), ", __t_min / 1e3,                   \
                static_cast<int64_t>(NUM_LOOP_ITERS * 1e6 / __t_min));  \
        }                                                               \
        printf("\n");                                                   \
    } while (0);

template <class data_t, class len_t>
static inline void randint_inplace(data_t* data, len_t len,
    data_t min=std::numeric_limits<data_t>::min(),
    data_t max=std::numeric_limits<data_t>::max())
{
    assert(data != nullptr);
    assert(len > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> d(min, max);

    for (len_t i = 0; i < len; i++) {
        data[i] = static_cast<data_t>(d(gen));
    }
}
template <class data_t, class len_t>
static inline void rand_inplace(data_t* data, len_t len,
                                data_t min=0, data_t max=1)
{
    assert(data != nullptr);
    assert(len > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(min, max);

    for (len_t i = 0; i < len; i++) {
        data[i] = static_cast<data_t>(d(gen));
    }
}

template<class data_t>
static inline data_t* aligned_random_ints(int64_t len) {
    data_t* ptr = aligned_alloc<data_t>(len);
    randint_inplace(ptr, len);
    return ptr;
}

template<class data_t>
static inline data_t* aligned_random(int64_t len) {
    data_t* ptr = aligned_alloc<data_t>(len);
    rand_inplace(ptr, len);
    return ptr;
}

#endif
