

#ifdef BLAZE
    #include "test/external/catch.hpp"
    #include "test/quantize/test_bolt.hpp"
    #include "src/include/public.hpp" // for Bolt wrapper class
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/memory.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "catch.hpp"
    #include "test_bolt.hpp"
    #include "public.hpp" // for Bolt wrapper class
    #include "testing_utils.hpp"
    #include "debug_utils.hpp"
    #include "memory.hpp"
#endif

static constexpr int M = _M; // M value from header // TODO this is a hack

TEST_CASE("bolt_smoketest", "[mcq][bolt]") {
    BoltEncoder enc(M);
    // TODO instantiate bolt encoder object here
    // printf("done");
}

//template<class data_t>
//void check_centroids(RowMatrix<data_t> C_in, ColMatrix<data_t> C_out) {
template<class data_t, class OutMatrixT>
void check_centroids(const RowMatrix<data_t>& C_in, const OutMatrixT& C_out) {
    for (int m = 0; m < ncodebooks; m++) {
        auto cin_start_ptr = C_in.data() + m * codebook_sz;
        auto cout_start_ptr = C_out.data() + m * codebook_sz;
        for (int i = 0; i < ncentroids; i++) { // for each centroid
            for (int j = 0; j < subvect_len; j++) { // for each dim
                CAPTURE(m);
                CAPTURE(i);
                CAPTURE(j);
                auto cin_ptr = cin_start_ptr + (subvect_len * i) + j;
                auto cout_ptr = cout_start_ptr + (ncentroids * j) + i;
                REQUIRE(*cin_ptr == *cout_ptr);
            }
        }
    }
}

TEST_CASE("bolt_encode_centroids", "[mcq][bolt]") {
    auto C = create_rowmajor_centroids<int>();
    ColMatrix<int> C_out(ncentroids, total_len);
    bolt_encode_centroids<M>(C.data(), total_len, C_out.data());

    check_centroids(C, C_out);

    SECTION("wrapper") {
        BoltEncoder enc(M);
        RowMatrix<float> centroids_float = C.cast<float>();
        enc.set_centroids(centroids_float.data(), C.rows(), C.cols());

        check_centroids(centroids_float, enc.centroids());
//        ColMatrix<float> centroids_col(enc.centroids());
//        check_centroids(centroids_float, centroids_col);
    }
}

void check_lut(const RowMatrix<float>& centroids_rowmajor,
    const RowVector<float>& q, const ColMatrix<uint8_t>& lut_out)
{
    for (int m = 0; m < ncodebooks; m++) {
        for (int i = 0; i < ncentroids; i++) {
            int dist_sq = 0;
            for (int j = 0; j < subvect_len; j++) {
                auto col = m * subvect_len + j;
                auto diff = centroids_rowmajor(i + m * ncentroids, j) - q(m * subvect_len + j);
                dist_sq += static_cast<int>(diff * diff);
            }
            CAPTURE(m);
            CAPTURE(i);
            REQUIRE(dist_sq == lut_out(i, m));
        }
    }
}

void _compute_lut(const float* q, int len, int nbytes,
                const RowMatrix<float>& centroids, const RowVector<float>& offsets,
                float scaleby, ColMatrix<uint8_t>& lut_out) {

    // ColMatrix<uint8_t>
    auto ncodebooks = 2 * nbytes;
    auto ncentroids = 16;
    auto subvect_len = centroids.cols();
    auto codebook_sz = subvect_len * ncentroids;
    for (int m = 0; m < ncodebooks; m++) {
        const float* block_ptr = centroids.data() + m * codebook_sz;
        for (int i = 0; i < ncentroids; i++) {
            float dist_sq = 0;
            for (int j = 0; j < subvect_len; j++) {
                auto col = m * subvect_len + j;
                auto diff = block_ptr[j * ncentroids + i] - q[col];
                dist_sq += diff * diff;
            }
            int dist_int = static_cast<int>(dist_sq * scaleby + offsets(m));
            lut_out(i, m) = std::max(0, std::min(255, dist_int));
            // dist_sq = std::fmax(0., std::fmin(dist_sq, 255.));
            // lut_out(i, m) = static_cast<uint8_t>(dist_sq);
        }
    }
}

void check_offset_scaled_lut(const RowVector<float>& q, int nbytes,
                             const RowMatrix<float>& encoded_centroids,
                             const RowVector<float>& offsets,
                             float scaleby,
                             const ColMatrix<uint8_t>& lut) {
    ColMatrix<uint8_t> lut_true(lut.rows(), lut.cols());
    _compute_lut(q.data(), (int)q.size(), nbytes, encoded_centroids,
               offsets, scaleby, lut_true);

//    std::cout << "lut_true:\n" << lut_true.cast<uint16_t>() << "\n";
//    std::cout << "---- lut:\n";
//    std::cout << lut.cast<uint16_t>() << "\n";

    int ncodebooks = 2 * nbytes;
    int ncentroids = 16;
    for (int j = 0; j < ncodebooks; j++) {
        for (int i = 0; i < ncentroids; i++) {
            auto diff = lut(i, j) - lut_true(i, j);
            CAPTURE((int)lut_true(i, j));
            CAPTURE((int)lut(i, j));
            REQUIRE(std::abs(diff) <= 1);
        }
    }
}


TEST_CASE("bolt_lut_l2", "[mcq][bolt]") {
    // create centroids with predictable patterns; note that we have to
    // be careful not to saturate the range of the uint8_t distances, which
    // means all of these entries have to within 15 of the corresponding
    // element of the query
    auto centroids_rowmajor = create_rowmajor_centroids<float>(1);
    ColMatrix<float> centroids(ncentroids, total_len);
    bolt_encode_centroids<M>(centroids_rowmajor.data(), total_len, centroids.data());

    RowVector<float> q = create_bolt_query();

    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);
    //    ColMatrix<float> lut_out(ncentroids, ncodebooks);
    //    ColMatrix<int32_t> lut_out(ncentroids, ncodebooks);
    //    ColMatrix<uint16_t> lut_out(ncentroids, ncodebooks);
    //    lut_out.fill(42); // there should be none of these when we print it
    bolt_lut<M>(q.data(), total_len, centroids.data(), lut_out.data());

//    std::cout << centroids_rowmajor << "\n\n";
//    std::cout << centroids << "\n\n";
//    std::cout << q << "\n";
//    std::cout << lut_out.cast<int>() << "\n";

    check_lut(centroids_rowmajor, q, lut_out);

    // for (int m = 0; m < ncodebooks; m++) {
    //     for (int i = 0; i < ncentroids; i++) {
    //         float dist_sq = 0;
    //         for (int j = 0; j < subvect_len; j++) {
    //             auto col = m * subvect_len + j;
    //             auto diff = centroids_rowmajor(i + m * ncentroids, j) - q(m * subvect_len + j);
    //             dist_sq += diff * diff;
    //         }
    //         CAPTURE(m);
    //         CAPTURE(i);
    //         REQUIRE(dist_sq == lut_out(i, m));
    //     }
    // }

    SECTION("wrapper") {
//        std::cout << "checking wrapper lut\n";  // TODO rm
        BoltEncoder enc(M);
        enc.set_centroids(centroids_rowmajor.data(), centroids_rowmajor.rows(),
                          centroids_rowmajor.cols());
        RowVector<float> offsets(2 * M);

        offsets.setZero();
        float scaleby = 1.0;
        enc.set_offsets(offsets.data(), (int)offsets.size());
        enc.set_scale(scaleby);

        SECTION("0 offset, scale = 1") {
            enc.lut_l2(q);
            auto lut = enc.get_lut(); // TODO func that returns it directly
            check_lut(centroids_rowmajor, q, lut);
            check_offset_scaled_lut(q, M, enc.centroids(), offsets, scaleby, lut);
        }

//        enc.set_offsets(offsets.data(), (int)offsets.size());
        SECTION("0 offset, scale = 2") {
            scaleby = 2.0;
            enc.set_scale(scaleby);
            enc.lut_l2(q);
            check_offset_scaled_lut(q, M, enc.centroids(), offsets, scaleby, enc.get_lut());
        }

        SECTION("random query, centroids, and offsets; scale = .7") {
            q.setRandom();
            RowVector<float> tmp(q);
            tmp.setRandom();
            q = (q.array() + tmp.array()).matrix();
            centroids_rowmajor.setRandom();
            enc.set_centroids(centroids_rowmajor.data(), centroids_rowmajor.rows(),
                              centroids_rowmajor.cols());
            offsets.setRandom();
            offsets *= 20;
            offsets = (offsets.array() + 20).matrix();
            scaleby = .7;
            enc.set_scale(scaleby);
            enc.set_offsets(offsets.data(), (int)offsets.size());
            enc.lut_l2(q);
            check_offset_scaled_lut(q, M, enc.centroids(), offsets, scaleby, enc.get_lut());
        }
    }
}

void check_encoding(int nrows, const ColMatrix<uint8_t>& encoding_out) {

    // number of rows in output must be a multiple of 32, and >32 would
    // require us to rewrite this to look at indices within each block
    // instead of letting column-major indexing handle everything for us
    size_t nrows_out = encoding_out.rows();
    REQUIRE(nrows_out == 32);

    for (int i = 0; i < nrows; i++) {
        for(int m = 0; m < 2 * M; m++) {
            // indices are packed into upper and lower 4 bits
            int byte = encoding_out(i, m / 2);
            int idx = m % 2 ? byte >> 4 : byte & 0x0F;
            REQUIRE(idx == m + (i % 5)); // i % 5 from how we designed mat
        }
    }
}

TEST_CASE("bolt_encode", "[mcq][bolt]") {
    auto centroids = create_bolt_centroids(1);

    SECTION("encode one vector") {
        // for 4 codebooks, subvect_len = 3, q =
        // [0, 1, 2, 18, 19, 20, 36, 37, 38, 54, 55, 56]
        RowVector<float> q(total_len);
        for (int m = 0; m < ncodebooks; m++) {
            for (int j = 0; j < subvect_len; j++) {
                auto idx = m * subvect_len + j;
                // add on a 2m at the end so which centroid it is changes by
                // 2 for each codebook
                q(idx) = ncentroids * m + j + (2 * m);
            }
        }

//        RowVector<uint8_t> encoding_out(M);
        ColMatrix<uint8_t> encoding_out(32, M); // enough space for whole block
        bolt_encode<M>(q.data(), 1, total_len, centroids.data(), encoding_out.data());

//        std::cout << "q: " << q << "\n";
//        std::cout << "centroids:\n" << centroids << "\n";
//        std::cout << "raw encoding bytes: " << encoding_out.cast<int>() << "\n";
//        std::cout << "encoding:\n";

        for (int m = 0; m < 2 * M; m++) {
            int byte = encoding_out(0, m / 2); // encodings are in first row
            int idx = m % 2 ? byte >> 4 : byte & 0x0F;
            REQUIRE(idx == 2 * m);
        }
//        std::cout << "\n";
    }

    SECTION("encode rows of matrix") {
        static constexpr int nrows = 10;
        auto X = create_X_matrix(nrows);
        ColMatrix<uint8_t> encoding_out(32, M);
        encoding_out.setZero(); // TODO rm after debug
        bolt_encode<M>(X.data(), nrows, total_len, centroids.data(),
                       encoding_out.data());

        check_encoding(nrows, encoding_out);

        SECTION("wrapper") {
            BoltEncoder enc(M);
            RowMatrix<float> centroids_rowmajor =
                create_rowmajor_centroids(1).cast<float>();
            enc.set_centroids(centroids_rowmajor.data(),
                centroids_rowmajor.rows(), centroids_rowmajor.cols());
            enc.set_data(X.data(), nrows, total_len);

//            PRINTLN_VAR(encoding_out.cast<int>());
//            PRINTLN_VAR(enc.codes().cast<int>());
            RowMatrix<uint8_t> codes_rowmajor(enc.codes());
            Map<ColMatrix<uint8_t> > codes_colmajor(codes_rowmajor.data(), 32, M);

            check_encoding(nrows, codes_colmajor);
        }
    }
}

TEST_CASE("bolt_scan", "[mcq][bolt]") {
    static constexpr int nblocks = 1; // arbitrary weird number
    static constexpr int nrows = 32 * nblocks;

    // create random codes from [0, 15]
    RowMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create centroids
    ColMatrix<float> centroids = create_bolt_centroids(1);

    // create query and look-up tables
    RowVector<float> q = create_bolt_query();
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    bolt_lut<M>(q.data(), total_len, centroids.data(), luts.data());

   // PRINTLN_VAR(codes.cast<int>());
//    PRINTLN_VAR(codes.topRows(2).cast<int>());
//    PRINTLN_VAR(centroids);
   // PRINTLN_VAR(luts.cast<int>());
//    PRINTLN_VAR(q);

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);
    bolt_scan<M>(codes.data(), luts.data(), dists_u8.data(), nblocks);
    bolt_scan<M>(codes.data(), luts.data(), dists_u16.data(), nblocks);
    bolt_scan<M, true>(codes.data(), luts.data(), dists_u16_safe.data(), nblocks);

   // PRINTLN_VAR(dists_u8.cast<int>());
   // PRINTLN_VAR(dists_u16.cast<int>());
   // PRINTLN_VAR(dists_u16_safe.cast<int>());

    check_bolt_scan(dists_u8.data(), dists_u16.data(), dists_u16_safe.data(),
                    luts, codes, M, nblocks);

    SECTION("wrapper") {
        BoltEncoder enc(M);
        RowMatrix<float> centroids = create_rowmajor_centroids(1).cast<float>();
        enc.set_centroids(centroids.data(), centroids.rows(), centroids.cols());
        enc.set_codes(codes); // subtlety: codes needs to be a RowMatrix

        auto dists = enc.dists_sq(q.data(), (int)q.size());

        // just replace safe dists16, since this is what wrapper uses
        check_bolt_scan(dists_u8.data(), dists_u16.data(), dists.data(),
                    luts, codes, M, nblocks);

//        printf("checked bolt wrapper scan\n");  // TODO rm
    }
}
