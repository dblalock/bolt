

#include "test_bolt.hpp"
#include "public.hpp" // for Bolt wrapper class

#include "testing_utils.hpp"
#include "debug_utils.hpp"
#include "memory.hpp"

static constexpr int M = _M; // M value from header // TODO this is a hack

TEST_CASE("bolt_smoketest", "[mcq][bolt]") {
    BoltEncoder enc(M);
    // TODO instantiate bolt encoder object here
    // printf("done");
}

template<class data_t>
void check_centroids(RowMatrix<data_t> C_in, ColMatrix<data_t> C_out) {
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
    }
}

void check_lut(const RowMatrix<float>& centroids_rowmajor,
    const RowVector<float>& q, const ColMatrix<uint8_t>& lut_out)
{
    for (int m = 0; m < ncodebooks; m++) {
        for (int i = 0; i < ncentroids; i++) {
            float dist_sq = 0;
            for (int j = 0; j < subvect_len; j++) {
                auto col = m * subvect_len + j;
                auto diff = centroids_rowmajor(i + m * ncentroids, j) - q(m * subvect_len + j);
                dist_sq += diff * diff;
            }
            CAPTURE(m);
            CAPTURE(i);
            REQUIRE(dist_sq == lut_out(i, m));
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
        BoltEncoder enc(M);
        enc.set_centroids(centroids_rowmajor.data(), centroids_rowmajor.rows(),
                          centroids_rowmajor.cols());
        auto lut = enc.lut(q);
        check_lut(centroids_rowmajor, q, lut);
    }
}

void check_encoding(int nrows, const RowMatrix<uint8_t>& encoding_out) {
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

        RowVector<uint8_t> encoding_out(M);
        bolt_encode<M>(q.data(), 1, total_len, centroids.data(), encoding_out.data());

//        std::cout << "q: " << q << "\n";
//        std::cout << "centroids:\n" << centroids << "\n";
//        std::cout << "raw encoding bytes: " << encoding_out.cast<int>() << "\n";
//        std::cout << "encoding:\n";

        for (int m = 0; m < 2 * M; m++) {
            int byte = encoding_out(m / 2);
            int idx = m % 2 ? byte >> 4 : byte & 0x0F;
            REQUIRE(idx == 2 * m);
        }
//        std::cout << "\n";
    }

    SECTION("encode rows of matrix") {
        static constexpr int nrows = 10;
        auto X = create_X_matrix(nrows);
        RowMatrix<uint8_t> encoding_out(nrows, M);
        bolt_encode<M>(X.data(), nrows, total_len, centroids.data(),
                       encoding_out.data());

        check_encoding(nrows, encoding_out);
        // for (int i = 0; i < nrows; i++) {
        //     for(int m = 0; m < 2 * M; m++) {
        //         // indices are packed into upper and lower 4 bits
        //         int byte = encoding_out(i, m / 2);
        //         int idx = m % 2 ? byte >> 4 : byte & 0x0F;
        //         REQUIRE(idx == m + (i % 5)); // i % 5 from how we designed mat
        //     }
        // }

        SECTION("wrapper") {
            BoltEncoder enc(M);
            RowMatrix<float> centroids_rowmajor = create_rowmajor_centroids(1).cast<float>();
            enc.set_centroids(centroids_rowmajor.data(),
                centroids_rowmajor.rows(), centroids_rowmajor.cols());
//            enc.set_data(X.data(), (int)X.rows(), (int)X.cols());
            enc.set_data(X.data(), nrows, total_len);

            // PRINTLN_VAR(encoding_out.cast<int>());
            // PRINTLN_VAR(enc.codes().cast<int>());

            check_encoding(nrows, enc.codes());
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

        auto dists = enc.dists_l2(q.data(), (int)q.size());

        check_bolt_scan(dists_u8.data(), dists.data(), dists_u16_safe.data(),
                    luts, codes, M, nblocks);
    }
}
