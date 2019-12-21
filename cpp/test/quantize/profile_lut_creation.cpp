//
//  profile_lut_creation.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifdef BLAZE
    #include "src/quantize/bolt.hpp"
    #include "src/quantize/product_quantize.hpp"
    #include "test/quantize/amm_common.hpp"
#else
    #include "bolt.hpp"
    #include "amm_common.hpp"
    #include "product_quantize.hpp"
#endif


void _profile_lut_mithral(int nrows, int ncols, int nbytes,
                          float lut_work_const)
                          // std::vector<float> lut_work_consts)
{
    int ncodebooks = nbytes * 2;
    // compute nnz to use for idxs
    // float max_const = lut_work_consts[0];
    // for (int i = 0; i < lut_work_consts.size(); i++) {
    //     max_const = MAX(max_const, lut_work_consts[i]);
    // }
    // int nnz = max_const > 0 ? ncols * lut_work_const : ncols;
    int nnz = lut_work_const > 0 ? ncols * (lut_work_const / ncodebooks): ncols;
    // printf("nnz: %d\n", nnz);

    ColMatrix<float> centroids16(ncodebooks*16, ncols); centroids16.setRandom();
    RowMatrix<float> X(nrows, ncols);                   X.setRandom();
    RowMatrix<uint8_t> lut_out(nrows, ncodebooks * 16); lut_out.setRandom();
    RowMatrix<float> lut_f32_out(nrows, ncodebooks*16); lut_f32_out.setRandom();
    RowVector<float> offsets(ncodebooks);               offsets.setRandom();
    // creating random sorted idxs for sparse centroids is a pain
    RowMatrix<int> idxs(ncodebooks, nnz);
    int all_idxs[ncols];
    for (int i = 0; i < ncols; i++) {
        all_idxs[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());  // why can't shuffle just create its own...
    for (int c = 0; c < ncodebooks; c++) {  // random sequential idxs
        std::shuffle(all_idxs, all_idxs + ncols, g);
        std::sort(all_idxs, all_idxs + nnz);
        for (int j = 0; j < nnz; j++) {
            idxs(c, j) = all_idxs[j];
        }
    }

    float offset = 0.;
    float scale = 1.;

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%%-20s, N D C B lut_work_const:, %7d, %3d, %3d, %2d, %%4.1f,\t",
        nrows, ncols, ncodebooks, nbytes, lut_work_const);
    auto fmt = fmt_as_cppstring.c_str();

    // ------------------------ mithral
    if (lut_work_const < 0) {
        msg = string_with_format(fmt, "mithral lut dense", -1.f);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            lut_out.data(), lut_out.size(),
            (mithral_lut_dense(X.data(), nrows, ncols, ncodebooks,
                centroids16.data(), offset, scale,
                lut_f32_out.data(), lut_out.data())) );
    } else {
        msg = string_with_format(fmt, "mithral lut sparse", lut_work_const);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        lut_out.data(), lut_out.size(),
        (mithral_lut_sparse(X.data(), nrows, ncols, ncodebooks,
            centroids16.data(), idxs.data(), nnz, offset, scale,
            lut_f32_out.data(), lut_out.data())) );
    }
}

void _profile_lut_others(int nrows, int ncols, int nbytes) {
    // if ((ncols < ncodebooks) || (ncols % ncodebooks)) { return; }

    int ncodebooks16 = nbytes * 2;
    int ncodebooks256 = nbytes;

    // shared
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    // bolt-specific
    ColMatrix<float> centroids16(ncodebooks16*16, ncols);
    RowMatrix<uint8_t> lut_out16(nrows, ncodebooks16 * 16);
    RowVector<float> offsets(ncodebooks16);
    centroids16.setRandom();
    lut_out16.setRandom();
    offsets.setRandom();
    float scale = 1;
    // pq + opq
    ColMatrix<float> centroids256(ncodebooks256 * 256, ncols);
    RowMatrix<float> lut_out256(nrows, ncodebooks256 * 256);
    RowMatrix<float> R(ncols, ncols);
    RowMatrix<float> X_rotated(nrows, ncols);
    centroids256.setRandom();
    lut_out256.setRandom();
    R.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%%-20s, N D C B lut_work_const:, %7d, %4d, %%3d, %2d, -1,   \t",
        nrows, ncols, nbytes);
    auto fmt = fmt_as_cppstring.c_str();

    // ------------------------ bolt
    if ((ncodebooks16 <= ncols) && (ncols % ncodebooks16 == 0)) {
        msg = string_with_format(fmt, "bolt lut cheating", ncodebooks16);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            lut_out16.data(), lut_out16.size(),
            (bolt_lut<Reductions::DotProd>(X.data(), nrows, ncols,
                    centroids16.data(), ncodebooks16, lut_out16.data()) ) );
        msg = string_with_format(fmt, "bolt lut", ncodebooks16);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            lut_out16.data(), lut_out16.size(),
            (bolt_lut<Reductions::DotProd>(X.data(), nrows, ncols,
                centroids16.data(), ncodebooks16, offsets.data(), scale,
                lut_out16.data()) ));
    }

    // ------------------------ pq
    if ((ncodebooks256 <= ncols)  && (ncols % ncodebooks256 == 0)) {
        msg = string_with_format(fmt, "pq lut", ncodebooks256);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            lut_out256.data(), lut_out256.size(),
            (pq_lut_8b<Reductions::DotProd>(X.data(), nrows, ncols, ncodebooks256,
                    centroids256.data(), lut_out256.data()) ) );

        // ------------------------ opq
        msg = string_with_format(fmt, "opq lut", ncodebooks256);
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            lut_out256.data(), lut_out256.size(),
            (opq_lut_8b<Reductions::DotProd>(X, ncodebooks256,
                    centroids256.data(), R, X_rotated, lut_out256.data()) ) );
    }
}

TEST_CASE("vq lut timing", "[amm][lut][profile]") {
//    static constexpr int nrows = 10 * 1024;
    static constexpr int nrows = 1 << 14;

    printf("algo, _0, N, D, C, B, lut_work_const, "
           "_1, latency0, _2, latency1, _3, latency2, _4,"
           "latency3, _5, latency4, _6\n");

    // static constexpr bool profile_mithral = true;
    static constexpr bool profile_mithral = false;
    static constexpr bool profile_others = true;

    std::vector<int> all_ncols {32, 64, 128, 256, 512, 1024};
    // std::vector<int> all_ncodebooks {16, 32, 64};
    std::vector<int> all_nbytes {8, 16, 32, 64};
    std::vector<float> lut_work_consts {-1.f, 2.f, 4.f};
    // std::vector<float> lut_work_consts {-1};
    // std::vector<float> lut_work_consts {2.f};
    for (auto ncols : all_ncols) {
        // for (auto c : all_ncodebooks) {
        for (auto b : all_nbytes) {
            if (profile_others) {
                _profile_lut_others(nrows, ncols, b);
            }
            if (profile_mithral) {
                for (auto w : lut_work_consts) {
                    _profile_lut_mithral(nrows, ncols, b, w);
                }
            }
        }
    }
}
