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


template<int NReadCols=-1, int NWriteCols=-1>
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

    if (NReadCols > 0 && NWriteCols > 0) {
        // MAX is just that it compiles in the case that these are negative,
        // because it tries to instantiate the template regardless
        sgemm_colmajor_narrow_padded<MAX(1, NReadCols), MAX(1, NWriteCols)>(
            A.data(), B.data(), N, D, M, C.data());
    } else {
        sgemm_colmajor(A.data(), B.data(), N, D, M, C.data());
    }

    ColMatrix<float> ans(N, M);
    ans = A * B;

    // std::cout << "------------------------ A:\n";
    // std::cout << A << std::endl;
    // std::cout << "------------------------ B:\n";
    // std::cout << B << std::endl;
    // std::cout << "------------------------ ours:\n";
    // std::cout << C << std::endl;
    // // std::cout << C.bottomRows(16) << std::endl;
    // std::cout << "------------------------ answer:\n";
    // std::cout << ans << std::endl;
    // // std::cout << ans.bottomRows(16) << std::endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            CAPTURE(NReadCols);
            CAPTURE(NWriteCols);
            CAPTURE(N);
            CAPTURE(D);
            CAPTURE(M);
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
    // _test_sgemm_colmajor<-1, -1>(1, 5, 1, true);
    // _test_sgemm_colmajor<-1, -1>(1, 1, 7, true);
    // _test_sgemm_colmajor<-1, -1>(1, 4, 5, true);
    // _test_sgemm_colmajor<-1, -1>(1, 5, 3, true);
    // int N = 256 + 8;
    // // int N = 256 * 2;
    // int D = 4;
    // int M = 2;
    // _test_sgemm_colmajor<4, 2>(N, D, M, true);

    using ivec = std::vector<int>;

    SECTION("4x2") {
        ivec Ns {8, 16, 24, 32, 200, 400};
        ivec Ds {4, 8, 12, 16, 20};
        ivec Ms {2, 4, 6};
        _test_sgemm_colmajor_many<4, 2>(Ns, Ds, Ms);
    }
    SECTION("4x4") {
        ivec Ns {8, 16, 24, 32, 200, 400};
        ivec Ds {4, 8, 12, 16, 20};
        ivec Ms {4, 8, 12};
        _test_sgemm_colmajor_many<4, 4>(Ns, Ds, Ms);
    }

    SECTION("7x1") {
        ivec Ns {8, 16, 24, 32, 200, 400};
        ivec Ds {7, 14, 21};
        ivec Ms {1, 2, 3};
        _test_sgemm_colmajor_many<7, 1>(Ns, Ds, Ms);
    }
    SECTION("8x1") {
        ivec Ns {8, 16, 24, 32, 200, 400};
        ivec Ds {8, 16, 24};
        ivec Ms {1, 2, 3};
        _test_sgemm_colmajor_many<8, 1>(Ns, Ds, Ms);
    }
    SECTION("3x2") {
        ivec Ns {8, 16, 24, 32, 200, 400};
        ivec Ds {3, 6, 9, 12};
        ivec Ms {2, 4, 6};
        _test_sgemm_colmajor_many<3, 2>(Ns, Ds, Ms);
    }
    SECTION("3x3") {
        ivec Ns {8, 16, 24, 32, 256, 256 + 8};
        ivec Ds {3, 6, 9, 12};
        ivec Ms {3, 6, 9};
        _test_sgemm_colmajor_many<3, 3>(Ns, Ds, Ms);
    }
    SECTION("brute force many shapes") {
        for (int n = 1; n <= 512; n = (n * 3) - 1) {
            printf("brute force sgemm test, n = %d...\n", n);
            for (int d = 1; d < 20; d++) {
                for (int m = 0; m < 20; m++) {
                    _test_sgemm_colmajor<-1, -1>(n, d, m);
                }
            }
        }
    }
}

