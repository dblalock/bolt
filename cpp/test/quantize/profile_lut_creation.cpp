//
//  profile_lut_creation.cpp
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

TEST_CASE("vq lut timing", "[amm][lut][profile]") {
    // static constexpr int nrows = 1024*1000;
    // static constexpr int nrows = 128*1000;
    // static constexpr int nrows = 4096;
    // static constexpr int nrows = 24 * 1000;
    // static constexpr int nrows = 24 * 500;
    static constexpr int nrows = 24 * 100;
    // static constexpr int nrows = 24 * 10;
    // static constexpr int nrows = 24;
    // static constexpr int nrows = 6;
    // static constexpr int nrows = 128;
    // static constexpr int64_t nrows = 1;
    // static constexpr int ncols = 24 * 16;               // length of vectors
    // static constexpr int ncols = 12 * 16;               // length of vectors
    // static constexpr int ncols = 1024;               // length of vectors
    static constexpr int ncols = 128;               // length of vectors
    // static constexpr int ncols = 127;               // length of vectors
    // static constexpr int ncols = 32;               // length of vectors
    // static constexpr int ncols = 16;               // length of vectors
    // static constexpr int ncols = 8;               // length of vectors
    // static constexpr int ncols = 1024 * 1024;               // length of vectors
    static constexpr int bits_per_codebook = 4;
    // static constexpr int ncodebooks = 32;
    static constexpr int ncodebooks = 16;
    // static constexpr int ncodebooks = 12;
    // static constexpr int ncodebooks = 8;
    // static constexpr int ncodebooks = 4;
    // static constexpr int ncodebooks = 2;
    static constexpr int ncentroids = (1 << bits_per_codebook);
    // static constexpr int nbytes = ncodebooks / 2;
    // static constexpr int nnz = ncols;  // like 10-15% slower than dense; or
    static constexpr int nnz = ncols * (2.f / ncodebooks);
    printf("nnz: %d\n", nnz);

    ColMatrix<float> centroids(ncodebooks * ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    RowMatrix<uint8_t> lut_out(nrows, ncodebooks * ncentroids);
    lut_out.setRandom();

    RowMatrix<float> lut_f32_out(nrows, ncodebooks * ncentroids);
    lut_f32_out.setRandom();

    RowVector<float> offsets(ncodebooks);
    offsets.setRandom();

    RowMatrix<int> idxs(ncodebooks, nnz);
    // idxs.setRandom();
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

    // printf("lut_out size: %d\n", (int)lut_out.size());

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dummy lut ", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     _dummy_lut<ncodebooks>(X.data(), ncols,
    //         centroids.data(), lut_out.data()));

    float offset = 0.;
    float scale = 1.;

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt lut cheating ", kNtrials,
        lut_out.data(), lut_out.size(),
        (bolt_lut<Reductions::DotProd>(X.data(), nrows, ncols,
                centroids.data(), ncodebooks, lut_out.data()) ) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt lut          ", kNtrials,
        lut_out.data(), lut_out.size(),
        (bolt_lut<Reductions::DotProd>(X.data(), nrows, ncols, centroids.data(),
            ncodebooks, offsets.data(), scale, lut_out.data()) ) );

    // // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral lut    ", kNtrials,
    // //     lut_out.data(), lut_out.size(),
    // //     (mithral_lut_v1(X.data(), nrows, ncols, ncodebooks,
    // //         centroids.data(), lut_out.data())) );

    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 1", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 2", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts<2>(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 4", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts<4>(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral quant lut 8", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (mithral_quantize_luts<8>(lut_f32_out.data(), nrows, ncodebooks,
    //         offset, scale, lut_out.data())));

    //     // lut_out.data(), lut_out.size(),
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "lut dense       2,2", kNtrials,
    //     lut_f32_out.data(), lut_f32_out.size(),
    //     (mithral_lut_dense(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), offsets.data(), offset, scale,
    //         lut_f32_out.data(), lut_out.data())) );
    //     // lut_out.data(), lut_out.size(),
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral lut dense ", kNtrials,
        lut_out.data(), lut_out.size(),
        (mithral_lut_dense(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), offset, scale,
            lut_f32_out.data(), lut_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "mithral lut sparse", kNtrials,
        lut_out.data(), lut_out.size(),
        (mithral_lut_sparse(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, offset, scale,
            lut_f32_out.data(), lut_out.data())) );

    // SELF: pick up by putting wrapper funcs in a cpp file; what happens right
    // now is that if we uncomment these calls to the fused func below, the
    // performance gets cut in half (which makes no sense at all); put wrappers
    // in cpp file so they'll just get compiled once and this sort of craziness
    // won't happen



    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "fused   lut f32 2,2", kNtrials,
    //     lut_f32_out.data(), lut_f32_out.size(),
    //     (dense_lut_f32_fused<2,2>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), offsets.data(), offset, scale,
    //         lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "fused   lut f32 2,3", kNtrials,
    //     lut_f32_out.data(), lut_f32_out.size(),
    //     (dense_lut_f32_fused<2,3>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), offsets.data(), offset, scale,
    //         lut_f32_out.data())) );

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "just quantize lut 1", kNtrials,
        lut_out.data(), lut_out.size(),
        (quantize_luts<1>(lut_f32_out.data(), nrows, ncodebooks,
            offsets.data(), scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "just quantize lut 2", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (quantize_luts<2>(lut_f32_out.data(), nrows, ncodebooks,
    //         offsets.data(), scale, lut_out.data())));
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "just quantize lut 4", kNtrials,
    //     lut_out.data(), lut_out.size(),
    //     (quantize_luts<4>(lut_f32_out.data(), nrows, ncodebooks,
    //         offsets.data(), scale, lut_out.data())));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    1,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<1,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    1,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<1,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    1,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<1,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    2,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<2,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    2,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<2,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "sparse lut    2,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (sparse_lut_f32<2,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), idxs.data(), nnz, lut_f32_out.data())) );


    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 1,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<1,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 1,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<1,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 1,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<1,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,1", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,1>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,2", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,2>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,3", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,3>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 2,4", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32<2,4>(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32    ", kNtrials,
        lut_f32_out.data(), lut_f32_out.size(),
        (dense_lut_f32(X.data(), nrows, ncols, ncodebooks,
            centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,1", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,1>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,2", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,2>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,3", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,3>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "dense lut f32 3,4", kNtrials,
    // lut_out.data(), lut_out.size(),
    //     (dense_lut_f32<3,4>(X.data(), nrows, ncols, ncodebooks,
    //         centroids.data(), lut_f32_out.data())) );
}
