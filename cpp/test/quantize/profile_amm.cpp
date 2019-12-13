//
//  profile_amm.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifdef BLAZE
    #include "test/quantize/profile_amm.hpp"
#else
    #include "profile_amm.hpp"
#endif

TEST_CASE("amm mithral", "[amm][mithral][matmul][profile]") {
     std::vector<int> ncodebooks {4, 8, 16, 32, 64};

     float lut_work_const = 2;
     _profile_mithral<int8_t>(kCaltechTaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kCaltechTaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kCifar10TaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kCifar100TaskShape, ncodebooks, lut_work_const);
     _profile_mithral(kUcrTaskShape, ncodebooks, lut_work_const);
}

TEST_CASE("amm bolt", "[amm][bolt][matmul][profile]") {
     std::vector<int> ncodebooks {4, 8, 16, 32, 64};
     _profile_bolt_amm(kCaltechTaskShape, ncodebooks);
     _profile_bolt_amm(kCifar10TaskShape, ncodebooks);
     _profile_bolt_amm(kCifar100TaskShape, ncodebooks);
     _profile_bolt_amm(kUcrTaskShape, ncodebooks);
}

TEST_CASE("amm linear approx matmul", "[amm][matmul][linear][profile]") {
    int N, D, M;
    // std::vector<int> dvals {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    std::vector<int> dvals {2, 4, 8, 16, 32, 64, 128}; // TODO uncomment above

    _profile_matmul_methods(dvals, kCaltechTaskShape);
    _profile_matmul_methods(dvals, kCifar10TaskShape);
    _profile_matmul_methods(dvals, kCifar100TaskShape);
    _profile_matmul_methods(dvals, kUcrTaskShape);
}

TEST_CASE("amm osnap", "[amm][matmul][osnap][linear][profile]") {
    std::vector<int> dvals {2, 4, 8, 16, 32, 64, 128};
    std::vector<int> nsketches {1, 2, 4};

    _profile_osnap(dvals, nsketches, kCaltechTaskShape);
    _profile_osnap(dvals, nsketches, kCifar10TaskShape);
    _profile_osnap(dvals, nsketches, kCifar100TaskShape);
    _profile_osnap(dvals, nsketches, kUcrTaskShape);
}
