//
//  main.cpp
//  Dig
//
//  Created by DB on 1/20/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#include <stdio.h>

// unit tests magic (keep this uncommented even if not doing catch run,
// or else test_* files will have missing symbols)
#define CATCH_CONFIG_RUNNER

#ifdef BLAZE
    #include "test/external/catch.hpp"
#else
    #include "catch.hpp"
#endif

int main(int argc, char *const argv[]) {
    return Catch::Session().run(argc, argv);
}

//#include <stdio.h>
//#include <string>
//#include <vector>
//
//
//#include "catch.hpp"
//#include "bolt.hpp"
////#include "mithral.hpp"
////#include "multisplit.hpp"
//#include "debug_utils.hpp"
//#include "eigen_utils.hpp"
//#include "timing_utils.hpp"
//#include "testing_utils.hpp"
//
//static constexpr int kNreps = 3;
//// static constexpr int kNreps = 1;
//static constexpr int kNtrials = 5;
//
//int main(int argc, char *const argv[]) {
//    static constexpr int nblocks = 64 * 1000;
//    static constexpr int nrows = nblocks * 32;
//    // static constexpr int ncodebooks = 64;
//    static constexpr int ncodebooks = 16;
//    // static constexpr int ncodebooks = 8;
//    // static constexpr int ncodebooks = 4;
//    static constexpr int ncentroids = 16;
//    static constexpr int M = ncodebooks / 2;
//
//    // create random codes from in [0, 15]
//    ColMatrix<uint8_t> codes(nrows, ncodebooks);
//    codes.setRandom();
//    codes = codes.unaryExpr(
//        [=](const uint8_t x) { return (uint8_t)(x % ncentroids); });
//
//    // create random luts
//    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
//    luts.setRandom();
//    luts = luts.array() / ncodebooks; // make max lut value small
//
//    ColMatrix<int8_t> luts_signed(ncentroids, ncodebooks);
//    luts_signed.setRandom();
//    luts_signed = luts_signed.array() / ncodebooks; // make max lut value small
//
//    // ColMatrix<int8_t> luts_i8(ncentroids, ncodebooks);
//    // luts_i8.setRandom();
//    // luts_i8 = luts_i8.array() / ncodebooks;
//
//    // do the scan to compute the distances
//    RowVector<uint8_t> dists_u8(nrows);
//    RowVector<uint16_t> dists_u16(nrows);
//    RowVector<uint16_t> dists_u16_safe(nrows);
//    // RowVector<int16_t> dists_u16_colmajor(nrows);
//    // RowVector<int16_t> dists_u16_colmajor_tile4(nrows);
//    RowVector<int16_t> dists_u16_colmajor_mithral(nrows);
//
//    // static constexpr int signed_luts = true;
//    static constexpr int signed_luts = false;
//
//    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint8                  ", kNtrials,
//        dists_u8.data(), nrows,
//        (bolt_scan<M, true, signed_luts>(
//            codes.data(), luts.data(), dists_u8.data(), nblocks)));
//
//    RowVector<uint8_t> dists_u8_x2(nrows * 2); // incase decides to upcast
//    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=4           ", kNtrials,
//        dists_u8_x2.data(), nrows,
//        (bolt_scan_avg(
//            codes.data(), luts.data(), dists_u8_x2.data(), nblocks, ncodebooks)));
//
////    return Catch::Session().run(argc, argv);
//}
