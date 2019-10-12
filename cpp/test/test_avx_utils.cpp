//
//  test_avx_utils.cpp
//  Bolt
//
//  Created by DB on 10/11/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <vector>

#ifdef BLAZE
    #include "src/utils/avx_utils.hpp"
    #include "test/external/catch.hpp"
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/eigen_utils.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "avx_utils.hpp"
    #include "catch.hpp"
    #include "debug_utils.hpp"
    #include "eigen_utils.hpp"
    #include "testing_utils.hpp"
#endif


template<int NReadCols=4, int NWriteCols=2>
void _test_sgemm_colmajor(int N, int D, int M, bool simple_entries=false) {
    ColMatrix<float> A(N, D);
    ColMatrix<float> B(D, M);
    ColMatrix<float> C(N, M);
    if (simple_entries) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                A(i, j) = 10 * j + i;
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < M; j++) {
                B(i, j) = 10 * j + i;
            }
        }
    } else {
        A.setRandom();
        B.setRandom();
    }
    C = (C.array() + -999).matrix();  // value we won't accidentally get

    sgemm_colmajor_narrow_padded<NReadCols, NWriteCols>(
        A.data(), B.data(), N, D, M, C.data());

    ColMatrix<float> ans(N, M);
    ans = A * B;

    // std::cout << "------------------------ ours:\n";
    // std::cout << C << std::endl;
    // std::cout << "------------------------ answer:\n";
    // std::cout << ans << std::endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            CAPTURE(i);
            CAPTURE(j);
            CAPTURE(C(i, j));
            CAPTURE(ans(i, j));
            REQUIRE(abs(C(i, j) - ans(i, j)) < .0001);
        }
    }
}

template<int NReadCols=4, int NWriteCols=2>
void _test_sgemm_colmajor_many(std::vector<int> Ns, std::vector<int> Ds,
    std::vector<int> Ms)
{
    for (auto N : Ns) {
        for (auto D : Ds) {
            for (auto M : Ms) {
                _test_sgemm_colmajor<NReadCols, NWriteCols>(N, D, M);
            }
        }
    }
}

TEST_CASE("sgemm colmajor", "[utils]") {
    int N = 16;
    int D = 8;
    int M = 4;
    _test_sgemm_colmajor(N, D, M, true);
    _test_sgemm_colmajor(N, D, M);
    _test_sgemm_colmajor(8, D, M);
    _test_sgemm_colmajor(32, D, M);
    _test_sgemm_colmajor(8, 4, 2);

    using ivec = std::vector<int>;

    SECTION("4x2") {
        ivec Ns {8, 16, 24, 32};
        ivec Ds {4, 8, 12, 16, 20};
        ivec Ms {2, 4, 6};
        _test_sgemm_colmajor_many<4, 2>(Ns, Ds, Ms);
    }
    SECTION("4x4") {
        ivec Ns {8, 16, 24, 32};
        ivec Ds {4, 8, 12, 16, 20};
        ivec Ms {4, 8, 12};
        _test_sgemm_colmajor_many<4, 4>(Ns, Ds, Ms);
    }

    SECTION("7x1") {
        ivec Ns {8, 16, 24, 32};
        ivec Ds {7, 14, 21};
        ivec Ms {1, 2, 3};
        _test_sgemm_colmajor_many<7, 1>(Ns, Ds, Ms);
    }
    SECTION("8x1") {
        ivec Ns {8, 16, 24, 32};
        ivec Ds {8, 16, 24};
        ivec Ms {1, 2, 3};
        _test_sgemm_colmajor_many<8, 1>(Ns, Ds, Ms);
    }
    SECTION("3x2") {
        ivec Ns {8, 16, 24, 32};
        ivec Ds {3, 6, 9, 12};
        ivec Ms {2, 4, 6};
        _test_sgemm_colmajor_many<3, 2>(Ns, Ds, Ms);
    }
    SECTION("3x3") {
        ivec Ns {8, 16, 24, 32};
        ivec Ds {3, 6, 9, 12};
        ivec Ms {3, 6, 9};
        _test_sgemm_colmajor_many<3, 3>(Ns, Ds, Ms);
    }
}

