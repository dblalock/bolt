//
//  profile_scan.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifdef BLAZE
    #include "test/quantize/amm_common.hpp"
#else
    #include "amm_common.hpp"
#endif

TEST_CASE("vq scan timing", "[scan][profile]") {
    static constexpr int nblocks = 64 * 1000;
    // static constexpr int nblocks = 2;
    // static constexpr int nblocks = 256;
    static constexpr int nrows = nblocks * 32;
    // static constexpr int ncodebooks = 64;
    // static constexpr int ncodebooks = 32;
    // static constexpr int ncodebooks = 24;
    static constexpr int ncodebooks = 16;
    // static constexpr int ncodebooks = 8;
    // static constexpr int ncodebooks = 4;
    static constexpr int ncentroids = 16;
    static constexpr int M = ncodebooks / 2;

    // create random codes from in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.unaryExpr(
        [=](const uint8_t x) { return (uint8_t)(x % ncentroids); });

    // create random luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    luts.setRandom();
    luts = luts.array() / ncodebooks; // make max lut value small

    ColMatrix<int8_t> luts_signed(ncentroids, ncodebooks);
    luts_signed.setRandom();
    luts_signed = luts_signed.array() / ncodebooks; // make max lut value small

    // ColMatrix<int8_t> luts_i8(ncentroids, ncodebooks);
    // luts_i8.setRandom();
    // luts_i8 = luts_i8.array() / ncodebooks;

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);
    // RowVector<int16_t> dists_u16_colmajor(nrows);
    // RowVector<int16_t> dists_u16_colmajor_tile4(nrows);
    RowVector<int16_t> dists_u16_colmajor_mithral(nrows);

    // static constexpr int signed_luts = true;
    static constexpr int signed_luts = false;

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint8                  ", kNtrials,
        dists_u8.data(), nrows,
        (bolt_scan<M, true, signed_luts>(
            codes.data(), luts.data(), dists_u8.data(), nblocks)));

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=4           ", kNtrials,
    //     dists_u8_x2.data(), nrows,
    //     (mithral_scan(
    //         codes.data(), luts.data(), dists_u8_x2.data(), nblocks, ncodebooks)));

    RowVector<uint8_t> dists_u8_x2(nrows * 2); // in case decides to upcast

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=2           ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 2>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=4           ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 4>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=8           ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 8>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan avg upcast=16          ", kNtrials,
        dists_u8_x2.data(), nrows,
        (mithral_scan<M, 16>(
            codes.data(), nblocks, luts.data(), dists_u8_x2.data())));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16                 ", kNtrials,
        dists_u16.data(), nrows,
        (bolt_scan<M, false, signed_luts>(
            codes.data(), luts.data(), dists_u16.data(), nblocks)));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16 safe            ", kNtrials,
        dists_u16_safe.data(), nrows,
        (bolt_scan<M, true, signed_luts>(
            codes.data(), luts.data(), dists_u16_safe.data(), nblocks)));
}
