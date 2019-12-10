//
//  test_mithral.cpp
//  Bolt
//
//  Created by DB on 10/28/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <stdio.h>

#ifdef BLAZE
    #include "test/external/catch.hpp"
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/memory.hpp"
    #include "src/quantize/mithral_v1.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "catch.hpp"
    #include "debug_utils.hpp"
    #include "memory.hpp"
    #include "mithral_v1.hpp"
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

void _zip4(const ColMatrix<uint8_t>& codes_in, ColMatrix<uint8_t>& out) {
    static const int in_block_sz = 16;
    int ncodebooks = (int)codes_in.cols();
    auto nblocks = codes_in.rows() / in_block_sz;
    assert(ncodebooks % 4 == 0);
    auto ncolstripes = ncodebooks / 4;

    // PRINT_VAR(ncodebooks);
    // PRINT_VAR(nblocks);
    // PRINT_VAR(ncolstripes);

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
// void _unzip4(const MatT0& codes_zipped, MatT1& out) {
//     assert(D % 4 == 0);

// }

ColMatrix<uint8_t> _unpack_low_hi(const ColMatrix<uint8_t>& codes_packed) {
    ColMatrix<uint8_t> out(codes_packed.rows(), 2 * codes_packed.cols());
    for (int i = 0; i < codes_packed.rows(); i++) {
        for (int j = 0; j < codes_packed.cols(); j++) {
            uint8_t val = codes_packed(i, j);
            auto j_out = 2 * j;
            out(i, j_out) = val & 0x0F;
            out(i, j_out + 1) = (val >> 4) & 0x0F;
        }
    }
    return out;
}



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
    _test_zip4(1, 8);
    _test_zip4(2, 4);
    _test_zip4(2, 8);
    _test_zip4(2, 12);
    _test_zip4(1, 20);
    _test_zip4(2, 24);
    _test_zip4(17, 24);
}


int8_t _lut_entry(int code, int codebook=0, int output=0) {
    // return code + codebook * 10 + output;
    // return (code + codebook - 10 * output) * (codebook % 5 ? 1 : -1);
    return (code + codebook - (10 * output));
}

template<int UpcastEvery=4>
void _test_mithral_scan_tiled(int nblocks, int ncodebooks, int nout=1) {
    static constexpr int ncentroids = 16;
    constexpr int block_nrows = 16; // always 16 in this file
    assert(ncodebooks % 4 == 0);
    // assert(N % 32 == 0);  // needed for zip func; or at least its current impl
    int N = nblocks * block_nrows;
    // assert(N % 16 == 0);
    // int N = nblocks * block_nrows;

    // create and populate luts
    Eigen::Tensor<int8_t, 3, Eigen::RowMajor> luts(nout, ncodebooks, ncentroids);
    for (int o = 0; o < nout; o++) {
        for (int c = 0; c < ncodebooks; c++) {
            for (int cc = 0; cc < ncentroids; cc++) {
                // luts(o, c, cc) = _lut_entry(cc, c, o) % 20; // TODO this fails
                luts(o, c, cc) = _lut_entry(cc, c, o);
                // luts(o, c, cc) = _lut_entry(cc, c);
                // luts(o, c, cc) = _lut_entry(cc, c, o) % 13;
                // luts(o, c, cc) = _lut_entry(cc, c, 3) % 20; // works
                // luts(o, c, cc) = _lut_entry(cc, c) % 20;
                // luts(o, c, cc) = _lut_entry(cc);
            }
        }
    }

    // for (int o = 0; o < nout; o++) {
    //     printf("luts for output col %d\n", o);
    //     // std::cout << "luts for output :";
    //     dump_elements(&luts(o, 0, 0), ncodebooks * ncentroids, ncentroids);
    //     // RowMatrix<uint8_t> tmp(luts.row(o))
    //     // std::cout << tmp.cast<int>() << "\n";
    // }
    // printf("raw luts mem:\n");
    // dump_elements(luts.data(), ncodebooks * ncentroids * nout, ncentroids);

    // create and populate codes
    ColMatrix<uint8_t> codes_unzipped(N, ncodebooks);
    codes_unzipped.setRandom();
    codes_unzipped = codes_unzipped.unaryExpr(
        [=](const uint8_t x) { return (uint8_t)(x % 16); });
    // for (int c = 0; c < ncodebooks; c++) {
    //     for (int n = 0; n < N; n++) {
    //         // codes_unzipped(n, c) = (n + 3 * c) % ncentroids;
    //         codes_unzipped(n, c) = (n + 2 * c) % ncentroids;
    //     }
    // }
    ColMatrix<uint8_t> codes_zipped(2 * N, ncodebooks / 4);
    _zip4(codes_unzipped, codes_zipped);

    // std::cout << "codes pre-zip:\n" << codes_unzipped.cast<int>() << "\n";

    // printf("zipped codes bytes:\n");
    // auto zipped_unpacked = _unpack_low_hi(codes_zipped);
    // std::cout << "unpacked zipped codes:\n" << zipped_unpacked.cast<int>() << "\n";

    // std::cout << "codes zipped:\n" << codes_zipped.cast<int>() << "\n";

    // TODO uncomment below

    // create and populate answers
    // ColVector<int16_t> ans(N);
    ColMatrix<int16_t> ans(N, nout);
    // Tensor<int16_t, 3> vals(N, ncodebooks);
    ColMatrix<int16_t> vals(N, ncodebooks);
    ans.setZero();
    for (int o = 0; o < nout; o++) {
        for (int n = 0; n < N; n++) {
            int16_t sum = 0;
            for (int c = 0; c < ncodebooks; c++) {
                auto code = codes_unzipped(n, c);
                auto val = luts(o, c, code);
                if (o == 0) { vals(n, c) = val; } // TODO rm
                sum += val;
            }
            ans(n, o) = sum;
        }
    }
    // std::cout << "intermediate vals:\n" << vals.cast<int>() << "\n";

    // get output from func
    // ColVector<int16_t> out(N, nout);
    ColMatrix<int16_t> out(N, nout);
    out.setZero();


    // <2, 2> fails with (1, 8, 2), but other amounts of tiling all work


    // mithral_scan<UpcastEvery>(codes_zipped.data(), N, ncodebooks, nout,
    mithral_scan_tiled<2>(codes_zipped.data(), N, ncodebooks, nout,
    // _mithral_scan<1, 1, UpcastEvery>(codes_zipped.data(), N, ncodebooks, nout,
    // _mithral_scan<1, 2, UpcastEvery>(codes_zipped.data(), N, ncodebooks, nout,
    // _mithral_scan<2, 1, UpcastEvery>(codes_zipped.data(), N, ncodebooks, nout,
    // _mithral_scan<2, 2, UpcastEvery>(codes_zipped.data(), N, ncodebooks, nout,
                              luts.data(), out.data());

    // if (nout == 1) {
    //     ColMatrix<int16_t>tmp(ans.rows(), 2);
    //     tmp.leftCols(1) = ans.rightCols(1);
    //     tmp.rightCols(1) = out.rightCols(1);
    //     std::cout << "ans vs out:\n" << tmp.cast<int>() << "\n";
    // } else if (nout == 2) {
    //     // left 2 cols are answers for the 2 outputs; right 2 cols are ours
    //     ColMatrix<int16_t>tmp(ans.rows(), 4);
    //     tmp.leftCols(2) = ans.rightCols(2);
    //     tmp.rightCols(2) = out.rightCols(2);
    //     std::cout << "ans vs out:\n" << tmp.cast<int>() << "\n";
    // }

    for (int o = 0; o < nout; o++) {
        for (int n = 0; n < ans.rows(); n++) {
            CAPTURE(o);
            CAPTURE(N);
            CAPTURE(ncodebooks);
            CAPTURE(n);
            CAPTURE((int)out(n, o));
            CAPTURE((int)ans(n, o));
            REQUIRE(abs(out(n, o) - ans(n, o)) < .0001);
        }
    }
}

TEST_CASE("mithral scan tiled", "[mithral][scan]") {

    SECTION("One output column") {
        _test_mithral_scan_tiled(1, 4);
        _test_mithral_scan_tiled(2, 4);
        _test_mithral_scan_tiled(3, 4);
        _test_mithral_scan_tiled<2>(1, 8);
        _test_mithral_scan_tiled<2>(2, 8);
        _test_mithral_scan_tiled<4>(2, 8);
        _test_mithral_scan_tiled<2>(2, 12);
        _test_mithral_scan_tiled<2>(7, 12);
        _test_mithral_scan_tiled<2>(3, 4 * 7);
    }

    SECTION("Multiple output columns") {
        _test_mithral_scan_tiled(1, 4, 2);
        _test_mithral_scan_tiled(1, 8, 2);
        _test_mithral_scan_tiled(2, 8, 2);
        _test_mithral_scan_tiled(2, 8, 3);
        _test_mithral_scan_tiled(2, 12, 3);
    }

    SECTION("Multiple chunks") {
        _test_mithral_scan_tiled(5 * 1024, 4, 1);
        _test_mithral_scan_tiled(5 * 1024, 4, 2);
        _test_mithral_scan_tiled(5 * 1024, 12, 2);
        _test_mithral_scan_tiled(5 * 1024, 12, 3);
    }
}
