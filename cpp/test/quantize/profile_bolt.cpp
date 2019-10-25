
// #include "test_bolt.hpp"

#include <string>

#ifdef BLAZE
    #include "test/external/catch.hpp"
    #include "src/quantize/bolt.hpp"
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/eigen_utils.hpp"
    #include "src/utils/timing_utils.hpp"
    #include "src/utils/memory.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "catch.hpp"
    #include "bolt.hpp"
    #include "debug_utils.hpp"
    #include "eigen_utils.hpp"
    #include "timing_utils.hpp"
    #include "testing_utils.hpp"
    #include "memory.hpp"
#endif

static constexpr int kNreps = 10;
// static constexpr int kNreps = 1;
static constexpr int kNtrials = 5;

static constexpr int ncols = 128;               // length of vectors
static constexpr int M = 8;                     // # bytes per compressed vect
static constexpr int64_t nrows_enc = 10*1000;   // number of rows to encode
static constexpr int64_t nrows_lut = 10*1000;   // number of luts to create
static constexpr int64_t nblocks_scan = 1000*1000 / 32;
static constexpr int64_t nblocks_query = 100*1000 / 32;
static constexpr int nqueries = 100;

static constexpr int bits_per_codebook = 4;
static constexpr int ncodebooks = M * (8 / bits_per_codebook);
static constexpr int ncentroids = (1 << bits_per_codebook);
static constexpr int ncentroids_total = ncentroids * ncodebooks;
static constexpr int lut_data_sz = ncentroids * ncodebooks;
static constexpr int subvect_len = ncols / ncodebooks;

static_assert(ncols % ncodebooks == 0,
    "ncols must be a multiple of ncodebooks!");

TEST_CASE("print bolt params", "[bolt][mcq][profile]") {
    printf("------------------------ bolt\n");
    printf("---- bolt profiling parameters\n");
    printf("bolt M: %d\n", M);
    printf("bolt nrows_enc: %g\n", (double)nrows_enc);
    printf("bolt nrows_lut: %g\n", (double)nrows_lut);
    printf("bolt nrows_scan: %g\n", (double)nblocks_scan * 32);
    printf("bolt nrows_query: %g\n", (double)nblocks_query * 32);
    printf("bolt subvect_len: %d\n", subvect_len);
    printf("bolt ncols: %d\n", ncols);
    printf("bolt nqueries: %d\n", nqueries);
    printf("---- bolt timings\n");
}

TEST_CASE("bolt encoding speed", "[bolt][encode][mcq][profile]") {
    static constexpr int nrows = nrows_enc;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    RowMatrix<uint8_t> encoding_out(nrows, M);

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt encode", kNtrials,
    encoding_out.data(), nrows * M,
        bolt_encode<M>(X.data(), nrows, ncols, centroids.data(),
                       encoding_out.data()));
}


TEST_CASE("bolt lut encoding speed", "[bolt][encode][mcq][profile]") {
    static constexpr int nrows = nrows_lut;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> Q(nrows, ncols);
    Q.setRandom();
    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);
    RowVector<float> offsets(ncols);
    offsets.setRandom();
    float scaleby = 3; // arbitrary number

    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt encode lut", kNtrials,
        lut_out.data(), lut_data_sz, nrows,
        bolt_lut<M>(Q.row(i).data(), ncols, centroids.data(),
            offsets.data(), scaleby, lut_out.data()));
}

TEST_CASE("bolt scan speed", "[bolt][scan][mcq][profile]") {
    static constexpr int nblocks = nblocks_scan;
    static constexpr int nrows = nblocks_scan * 32;

    // create random codes in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create random luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    luts.setRandom();
    luts = luts.array() / (2 * M); // make max lut value small

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint8", kNtrials,
        dists_u8.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u8.data(), nblocks));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16", kNtrials,
        dists_u16.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u16.data(), nblocks));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16 safe", kNtrials,
        dists_u16_safe.data(), nrows,
        bolt_scan<(M, true)>(
            codes.data(), luts.data(), dists_u16_safe.data(), nblocks));
}

template<int M, bool Safe=false, class dist_t=void>
void _run_query(const uint8_t* codes, int nblocks,
    const float* q, int ncols,
    const float* centroids,
    uint8_t* lut_out, dist_t* dists_out)
{
    bolt_lut<M>(q, ncols, centroids, lut_out);
    bolt_scan<M, Safe>(codes, lut_out, dists_out, nblocks);
}

TEST_CASE("bolt query (lut + scan) speed", "[bolt][mcq][profile]") {
    static constexpr int nblocks = nblocks_query;
    static constexpr int nrows = nblocks * 32;

    // create random codes from in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create random queries
    RowMatrix<float> Q(nqueries, ncols);
    Q.setRandom();

    // create random centroids
    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();

    // storage for luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);

    SECTION("uint8_t") {
        RowVector<uint8_t> dists(nrows);
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt query u8", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query<M>(codes.data(), nblocks, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) );
    }
    SECTION("uint16_t") {
        RowVector<uint16_t> dists(nrows);
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt query u16", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query<M>(codes.data(), nblocks, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) );
    }
    SECTION("uint16_t safe") {
        RowVector<uint16_t> dists(nrows);
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt query u16 safe", kNtrials,
            dists.data(), nrows, nqueries,
            (_run_query<M, true>(codes.data(), nblocks, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) ));
    }
}


template<int M, bool NeedEncodeX>
void _run_bolt_matmul(const RowMatrix<float>& X, const RowMatrix<float>& Q,
    ColMatrix<float> centroids, RowMatrix<uint8_t> codes,
    RowVector<float> offsets, float scaleby,
    ColMatrix<uint8_t> lut_out, RowMatrix<uint16_t> out)
{
    auto ncols = X.cols();
    auto nqueries = Q.rows();
    auto nblocks = X.rows() / 32;
    REQUIRE(ncols == Q.cols());
    REQUIRE((nblocks * 32) == X.rows());

    if (NeedEncodeX) {
        bolt_encode<M>(X.data(), X.rows(), (int)X.cols(), centroids.data(), codes.data());
    }

    for (int i = 0; i < nqueries; i++) {
        const float* q = Q.row(i).data();
        bolt_lut<M>(q, (int)ncols, centroids.data(), offsets.data(),
                    scaleby, lut_out.data());
        bolt_scan<M, true>(codes.data(), lut_out.data(), out.row(i).data(), nblocks);
    }
}

template<int M>
void _profile_bolt_matmul(int nrows, int ncols, int nqueries) {
    static constexpr int ncodebooks = 2 * M;

    // create random data
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();

    // create random queries
    RowMatrix<float> Q(nqueries, ncols);
    Q.setRandom();

    // create random centroids
    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();

    // create random codes in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create random / arbitrary offsets and scale factor for luts
    RowVector<float> offsets(ncols);
    offsets.setRandom();
    float scaleby = 3; // arbitrary number

    // storage for luts, product
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    RowMatrix<uint16_t> out(nqueries, nrows);

    // time it
    std::string msg = string_with_format("bolt<%d> encode=%d matmul %d",
                                         M, false, nqueries);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), nrows * nqueries,
        (_run_bolt_matmul<M, false>(X, Q, centroids, codes, offsets, scaleby, luts, out)) );

    std::string msg2 = string_with_format("bolt<%d> encode=%d matmul %d",
                                         M, true, nqueries);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg2, kNtrials,
        out.data(), nrows * nqueries,
        (_run_bolt_matmul<M, true>(X, Q, centroids, codes, offsets, scaleby, luts, out)) );
}

TEST_CASE("bolt square matmul speed", "[square][matmul][profile]") {

    // uncomment to profile square matrix multiplies
    std::vector<int> sizes {64, 128, 256};
    // std::vector<int> sizes {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    for (auto sz : sizes) {
       _profile_bolt_matmul<8>(sz, sz, sz);
       _profile_bolt_matmul<16>(sz, sz, sz);
       _profile_bolt_matmul<32>(sz, sz, sz);
    }
}
TEST_CASE("bolt tall matmul speed", "[tall][matmul][profile]") {
    // profile tall skinny matmuls; basically like answering mips queries
    static constexpr int nrows = 100 * 1000;
    static constexpr int ncols = 256;
    std::vector<int> nums_queries {1, 16, 32, 64};
    // std::vector<int> nums_queries {1, 16, 32, 64, 128, 256, 512, 1024, 2048};
    for (auto nqueries : nums_queries) {
        _profile_bolt_matmul<8>(nrows, ncols, nqueries);
        _profile_bolt_matmul<16>(nrows, ncols, nqueries);
        _profile_bolt_matmul<32>(nrows, ncols, nqueries);
    }
}

template<class MatrixT1, class MatrixT2, class MatrixT3>
void _run_matmul(const MatrixT1& X, const MatrixT2& Q, MatrixT3& out) {
   out.noalias() = X * Q;
}

void _profile_matmul(int nrows, int ncols, int nqueries) {
    // using MatrixT = ColMatrix<float>;
    using MatrixT = RowMatrix<float>; // faster for small batches, else slower

    // create random data
    MatrixT X(nrows, ncols);
    X.setRandom();

    // create random queries
    // RowMatrix<float> Q(ncols, nqueries);
    MatrixT Q(ncols, nqueries);
    Q.setRandom();

    // create output matrix to avoid malloc
    MatrixT out(nrows, nqueries);

    // time it
    std::string msg = string_with_format("matmul %d", nqueries);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), nrows * nqueries,
        _run_matmul(X, Q, out));
}


// TODO move this to own file; it's something we'd like to profile in general
TEST_CASE("square matmul speed", "[square][matmul][profile]") {
    // square matrix
    std::vector<int> sizes {64, 128, 256};
    // std::vector<int> sizes {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    for (auto sz : sizes) {
        _profile_matmul(sz, sz, sz);
    }
}
TEST_CASE("tall matmul speed", "[tall][matmul][profile]") {
    // tall skinnny matrix
    static constexpr int nrows = 100 * 1000;
    static constexpr int ncols = 256;
    std::vector<int> nums_queries {1, 16, 32, 64};
    // std::vector<int> nums_queries {1, 16, 32, 64, 128, 256, 512, 1024, 2048};
    for (auto nqueries : nums_queries) {
       _profile_matmul(nrows, ncols, nqueries);
    }
}
