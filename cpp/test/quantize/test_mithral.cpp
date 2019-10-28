//
//  test_mithral.cpp
//  Bolt
//
//  Created by DB on 10/28/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <stdio.h>

#ifdef BLAZE
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/memory.hpp"
    #include "src/quantize/mithral.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "debug_utils.hpp"
    #include "memory.hpp"
    #include "mithral.hpp"
    #include "testing_utils.hpp"
#endif


template<typename MatT0, typename MatT1>
void _zip4_one_block(const MatT0& codes_in, MatT1& out,
                     int in_row_offset=0, int out_row_offset=0,
                     int in_col_offset=0, int out_col_offset=0)
{
    int in_offset = in_row_offset;
    int out_offset = out_row_offset;
    int j_in = in_col_offset;
    int j_out = out_col_offset;
    int lut_sz = 16;
    // ac
    for (int i = 0; i < lut_sz; i++) {
        out(i + out_offset, j_out) = (
            codes_in(i + in_offset, 2 + j_in) << 4) +
            codes_in(i + in_offset, j_in);
    }
    // bd
    out_offset += lut_sz;
    for (int i = 0; i < lut_sz; i++) {
        out(i + out_offset, j_out) = (
            codes_in(i + in_offset, 3 + j_in) << 4) +
            codes_in(i + in_offset, 1 + j_in);
    }
    // // ef
    // in_offset += lut_sz;
    // out_offset += lut_sz;
    // for (int i = 0; i < lut_sz; i++) {
    //     out(i + out_offset, j_out) = (
    //         codes_in(i + in_offset, 2 + j_in) << 4) +
    //         codes_in(i + in_offset, 0 + j_in);
    // }
    // // gh
    // out_offset += lut_sz;
    // for (int i = 0; i < lut_sz; i++) {
    //     out(i + out_offset, j_out) = (
    //         codes_in(i + in_offset, 3 + j_in) << 4) +
    //         codes_in(i + in_offset, 1 + j_in);
    // }
}

template<typename MatT0, typename MatT1>
void _zip4(const MatT0& codes_in, MatT1& out) {
    static const int in_block_sz = 16;
    int ncodebooks = (int)codes_in.cols();
    int nblocks = codes_in.rows() / in_block_sz;
    assert(ncodebooks % 4 == 0);
    auto ncolstripes = ncodebooks / 4;

    // PRINT_VAR(ncodebooks);
    // PRINT_VAR(nblocks);

    for (int j = 0; j < ncolstripes; j++) {
        auto in_col_offset = 4 * j;
        auto out_col_offset = j;
        for (int b = 0; b < nblocks; b++) {
            auto in_row_offset = b * in_block_sz;
            auto out_row_offset = 2 * in_row_offset;
            // printf("(%2d, %2d) -> (%2d, %2d)\n", in_row_offset, in_col_offset,
            //     out_row_offset, out_col_offset);
            _zip4_one_block(codes_in, out, in_row_offset, out_row_offset,
                            in_col_offset, out_col_offset);
        }
    }
}

// template<typename MatT0, typename MatT1>
// void _unzip4_one_block(const MatT0& codes_zipped, MatT1& out,
//                        int in_row_offset=0, int out_row_offset=0,
//                        int in_col_offset=0, int out_col_offset=0)
// {


// }

// template<typename MatT0, typename MatT1>
// void _unzip4(const MatT0& codes_zipped, int D, int nblocks, MatT1& out) {
//     assert(D % 4 == 0);

// }



void _test_zip4(int nblocks, int ncodebooks) {
    nblocks *= 2; // simd zip4 func requires 32 codes, not 16
    // int in_block_sz = 32;
    int in_block_sz = 16;
    int out_block_sz = 16;
    int N = in_block_sz * nblocks;
    // assert(nblocks % 2 == 0); // our simd zip func requires this

    ColMatrix<uint8_t> codes_in(N, ncodebooks);
    codes_in.setRandom();
    codes_in = codes_in.unaryExpr(
        [=](const uint8_t x) { return (uint8_t)(x % 16); });
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < D; j++) {
    //         codes_in(i, j) = (i + D * j) % 16;
    //     }
    // }

    ColMatrix<uint8_t> codes_out(2 * N, ncodebooks / 4);
    ColMatrix<uint8_t> ans(2 * N, ncodebooks / 4);

    zip4_4b_colmajor(codes_in.data(), N, ncodebooks, codes_out.data());

    // _zip4(codes_in, ncodebooks, nblocks, ans);
    _zip4(codes_in, ans);

    // ColMatrix<uint8_t>tmp(ans.rows(), 2);
    // tmp.leftCols(1) = ans.rightCols(1);
    // tmp.rightCols(1) = codes_out.rightCols(1);
    // std::cout << "ans vs out:\n" << tmp.cast<int>() << "\n";

    // std::cout << "in:\n" << codes_in.cast<int>() << "\n";
    // std::cout << "ans:\n" << ans.cast<int>() << "\n";
    // std::cout << "out:\n" << codes_out.cast<int>() << "\n";
    // std::cout << "ans:\n" << ans.cast<int>().rightCols(1) << "\n";
    // std::cout << "out:\n" << codes_out.cast<int>().rightCols(1) << "\n";

    for (int i = 0; i < ans.rows(); i++) {
        for (int j = 0; j < ans.cols(); j++) {
            CAPTURE(N);
            CAPTURE(ncodebooks);
            CAPTURE(i);
            CAPTURE(j);
            CAPTURE((int)codes_out(i, j));
            CAPTURE((int)ans(i, j));
            REQUIRE(abs(codes_out(i, j) - ans(i, j)) < .0001);
        }
    }
}

TEST_CASE("mithral zip4", "[mithral][utils]") {
    _test_zip4(1, 4);
    // _test_zip4(1, 8);
    // _test_zip4(2, 4);
    // _test_zip4(2, 8);
    // _test_zip4(2, 12);
    // _test_zip4(1, 20);
    // _test_zip4(2, 24);
    // _test_zip4(17, 24);
}


int8_t _lut_entry(int code, int codebook=0, int output=0) {
    return code + codebook * 10 + output;
}

void _test_mithral_scan(int N, int ncodebooks, int nout=1) {
    static constexpr int ncentroids = 16;
    // static constexpr int block_nrows = 16;
    assert(ncodebooks % 4 == 0);
    // assert(N % 32 == 0);  // needed for zip func; or at least its current impl
    assert(N % 16 == 0);
    // int N = nblocks * block_nrows;

    // create and populate luts
    Eigen::Tensor<int8_t, 3, Eigen::RowMajor> luts(nout, ncodebooks, ncentroids);
    for (int o = 0; o < nout; o++) {
        for (int c = 0; c < ncodebooks; c++) {
            for (int cc = 0; cc < ncentroids; cc++) {
                luts(o, c, cc) = _lut_entry(cc, c);
            }
        }
    }

    // create and populate codes
    ColMatrix<uint8_t> codes_unzipped(N, ncodebooks);
    codes_unzipped.setRandom();
    codes_unzipped = codes_unzipped.unaryExpr(
        [=](const uint8_t x) { return (uint8_t)(x % 16); });
    ColMatrix<uint8_t> codes_zipped(2 * N, ncodebooks / 4);
    _zip4(codes_unzipped, codes_zipped);

    // create and populate answers
    ColVector<int16_t> ans(N);
    ans.setZero();
    for (int o = 0; o < nout; o++) {
        for (int n = 0; n < N; n++) {
            int16_t sum = 0;
            for (int c = 0; c < ncodebooks; c++) {
                auto code = codes_unzipped(n, c);
                sum += luts(o, c, code);
            }
            ans(n) = sum;
        }
    }

    // get output from func
    ColVector<int16_t> out(N);
    static constexpr int noutputs_per_stripe = 1;
    mithral_scan<(2, noutputs_per_stripe)>(
            codes_zipped.data(), N, ncodebooks,
            nout, luts.data(), out.data());

    for (int n = 0; n < ans.rows(); n++) {
            CAPTURE(N);
            CAPTURE(ncodebooks);
            CAPTURE(n);
            CAPTURE((int)out(n));
            CAPTURE((int)ans(n));
            REQUIRE(abs(out(n) - ans(n)) < .0001);
    }

    // luts_signed.setRandom();
    // luts_signed = luts_signed.array() / ncodebooks; // make max lut value small

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < D; j++) {
    //         codes_in(i, j) = (i + D * j) % 16;
    //     }
    // }
}

TEST_CASE("mithral scan", "[mithral][scan]") {
    // _test_mithral_scan(1, 4);
    // in wrapper func to take in different sizes
    // create random codes
    // create deterministic luts (no overflow, val = 10+code or something)
    //  make a simple func to avoid dup code when making ans and luts
    // create output buff + ans buff
    // populate ans buff based on codes
    // populate output buff via zip4 of codes, then calling scan func


    // SELF: pick up here


    printf("test mithral scan done\n");
}
