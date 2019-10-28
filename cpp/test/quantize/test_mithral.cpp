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

TEST_CASE("bolt_zip", "[mcq][bolt][utils]") {
    int nblocks = 1;
    int block_sz = 32;
    int N = block_sz * nblocks;
    int D = 4;
    ColMatrix<uint8_t> codes_in(N, D);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            codes_in(i, j) = (i + D * j) % 16;
        }
    }

    ColMatrix<uint8_t> codes_out(2 * N, D / 4);
    ColMatrix<uint8_t> ans(2 * N, D / 4);

    zip4_4b_colmajor(codes_in.data(), D, nblocks, codes_out.data());

    int in_offset = 0;
    int out_offset = 0;
    int lut_sz = 16;
    // ac
    for (int i = 0; i < lut_sz; i++) {
        ans(i + out_offset, 0) = (codes_in(i + in_offset, 2) << 4) + codes_in(i + in_offset, 0);
    }
    // bd
    out_offset += lut_sz;
    for (int i = 0; i < lut_sz; i++) {
        ans(i + out_offset, 0) = (codes_in(i + in_offset, 3) << 4) + codes_in(i + in_offset, 1);
    }
    // ef
    in_offset += lut_sz;
    out_offset += lut_sz;
    for (int i = 0; i < lut_sz; i++) {
        ans(i + out_offset, 0) = (codes_in(i + in_offset, 2) << 4) + codes_in(i + in_offset, 0);
    }
    // gh
    out_offset += lut_sz;
    for (int i = 0; i < lut_sz; i++) {
        ans(i + out_offset, 0) = (codes_in(i + in_offset, 3) << 4) + codes_in(i + in_offset, 1);
    }

    // std::cout << "in:\n" << codes_in.cast<int>() << "\n";
    // std::cout << "ans:\n" << ans.cast<int>() << "\n";
    // std::cout << "out:\n" << codes_out.cast<int>() << "\n";

    for (int i = 0; i < ans.rows(); i++) {
        for (int j = 0; j < ans.cols(); j++) {
            CAPTURE(N);
            CAPTURE(D);
            CAPTURE(i);
            CAPTURE(j);
            CAPTURE(codes_out(i, j));
            CAPTURE(ans(i, j));
            REQUIRE(abs(codes_out(i, j) - ans(i, j)) < .0001);
        }
    }
}
