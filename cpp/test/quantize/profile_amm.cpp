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

TEST_CASE("amm mithral", "[amm][matmul][mithral][profile]") {
     std::vector<int> ncodebooks {2, 4, 8, 16, 32, 64};
     std::vector<float> lutconsts {-1, 1, 2, 4};
     _profile_mithral<int8_t>(kCaltechTaskShape0, ncodebooks, lutconsts);
     _profile_mithral<int8_t>(kCaltechTaskShape1, ncodebooks, lutconsts);
     _profile_mithral(kCaltechTaskShape0, ncodebooks, lutconsts);
     _profile_mithral(kCaltechTaskShape1, ncodebooks, lutconsts);
     _profile_mithral(kCifar10TaskShape, ncodebooks, lutconsts);
     _profile_mithral(kCifar100TaskShape, ncodebooks, lutconsts);
     _profile_mithral(kUcrTaskShape0, ncodebooks, lutconsts);
     _profile_mithral(kUcrTaskShape1, ncodebooks, lutconsts);
     _profile_mithral(kUcrTaskShape2, ncodebooks, lutconsts);
}

TEST_CASE("amm bolt", "[amm][matmul][bolt][profile]") {
     std::vector<int> ncodebooks {2, 4, 8, 16, 32, 64};
     _profile_bolt_amm(kCaltechTaskShape0, ncodebooks);
     _profile_bolt_amm(kCaltechTaskShape1, ncodebooks);
     _profile_bolt_amm(kCifar10TaskShape, ncodebooks);
     _profile_bolt_amm(kCifar100TaskShape, ncodebooks);
     _profile_bolt_amm(kUcrTaskShape0, ncodebooks);
     _profile_bolt_amm(kUcrTaskShape1, ncodebooks);
     _profile_bolt_amm(kUcrTaskShape2, ncodebooks);
}

TEST_CASE("amm linear approx matmul", "[amm][matmul][dense][linear][profile]") {
    int N, D, M;
    // std::vector<int> dvals {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    std::vector<int> dvals {2, 4, 8, 16, 32, 64, 128}; // TODO uncomment above

    _profile_matmul_methods(dvals, kCaltechTaskShape0);
    _profile_matmul_methods(dvals, kCaltechTaskShape1);
    _profile_matmul_methods(dvals, kCifar10TaskShape);
    _profile_matmul_methods(dvals, kCifar100TaskShape);
    _profile_matmul_methods(dvals, kUcrTaskShape0);
    _profile_matmul_methods(dvals, kUcrTaskShape1);
    _profile_matmul_methods(dvals, kUcrTaskShape2);
}

TEST_CASE("amm osnap", "[amm][matmul][osnap][linear][profile]") {
    std::vector<int> dvals {2, 4, 8, 16, 32, 64, 128};
    std::vector<int> nsketches {1, 2, 4};

    _profile_osnap(dvals, nsketches, kCaltechTaskShape0);
    _profile_osnap(dvals, nsketches, kCaltechTaskShape1);
    _profile_osnap(dvals, nsketches, kCifar10TaskShape);
    _profile_osnap(dvals, nsketches, kCifar100TaskShape);
    _profile_osnap(dvals, nsketches, kUcrTaskShape0);
    _profile_osnap(dvals, nsketches, kUcrTaskShape1);
    _profile_osnap(dvals, nsketches, kUcrTaskShape2);
}

TEST_CASE("amm sparse", "[amm][matmul][sparse][linear][profile]") {
    std::vector<int> dvals {2, 4, 8, 16, 32, 64, 128};
    // std::vector<float> nnz_fracs(19);  // .05 thru .95
    // for (int i = 0; i < nnz_fracs.size(); i++) {
    //     nnz_fracs[i] = (i + 1) * .05;
    // }
    std::vector<float> nnz_fracs {
        .01, .025, .05, .1, .15, .2, .25, .3,
        .35, .4,   .45, .5, .6,  .7, .8,  .9,
        1.};

    _profile_sparse_amm(dvals, nnz_fracs, kCaltechTaskShape0);
    _profile_sparse_amm(dvals, nnz_fracs, kCaltechTaskShape1);
    _profile_sparse_amm(dvals, nnz_fracs, kCifar10TaskShape);
    _profile_sparse_amm(dvals, nnz_fracs, kCifar100TaskShape);
    _profile_sparse_amm(dvals, nnz_fracs, kUcrTaskShape0);
    _profile_sparse_amm(dvals, nnz_fracs, kUcrTaskShape1);
    _profile_sparse_amm(dvals, nnz_fracs, kUcrTaskShape2);
}
