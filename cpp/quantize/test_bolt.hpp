
#include "catch.hpp"

#include "bolt.hpp"
#include "eigen_utils.hpp"

namespace {

static constexpr int _M = 2;
static constexpr int subvect_len = 3;
static constexpr int ncodebooks = 2 * _M;
static constexpr int total_len = ncodebooks * subvect_len;
static constexpr int total_sz = 16 * total_len;
static constexpr int ncentroids = 16; // always 16 for 4 bits
static constexpr int ncentroids_total = ncentroids * ncodebooks;
static constexpr int codebook_sz = ncentroids * subvect_len;

template<class T>
static inline RowMatrix<T> create_rowmajor_centroids(T centroid_step=10,
    T codebook_step=16)
{
    RowMatrix<T> C(ncentroids_total, subvect_len);
    for (int i = 0; i < ncentroids_total; i++) {
        int centroid_start_val = centroid_step * (i % ncentroids) +
            (i / ncentroids) * codebook_step;
        for (int j = 0; j < subvect_len; j++) {
            C(i, j) = centroid_start_val + j;
        }
    }
    return C;
}

static inline ColMatrix<float> create_bolt_centroids(float centroid_step=1,
    float codebook_step=16)
{
    auto centroids_rowmajor = create_rowmajor_centroids<float>(
        centroid_step, codebook_step);
    ColMatrix<float> centroids(ncentroids, total_len);
    bolt_encode_centroids<_M>(centroids_rowmajor.data(), total_len, centroids.data());
    return centroids;
}

static inline RowVector<float> create_bolt_query() {
    // for 4 codebooks, subvect_len = 3, q =
    // [0, 1, 2, 18, 19, 20, 36, 37, 38, 54, 55, 56]
    RowVector<float> q(total_len);
    for (int m = 0; m < ncodebooks; m++) {
        for (int j = 0; j < subvect_len; j++) {
            auto idx = m * subvect_len + j;
            q(idx) = ncentroids * m + j + (ncentroids / 2);
        }
    }
    return q;
}

static inline RowMatrix<float> create_X_matrix(int64_t nrows) {
    RowMatrix<float> X(nrows, total_len);
    for (int i = 0; i < nrows; i++) {
        for (int m = 0; m < ncodebooks; m++) {
            for (int j = 0; j < subvect_len; j++) {
                auto idx = m * subvect_len + j;
                // add on m at the end so which centroid it is changes by
                // 1 for each codebook; also add on i so that each row
                // will pick centroids 1 higher the previous ones
                X(i, idx) = ncentroids * m + j + m + (i % 5);
            }
        }
    }
    return X;
}

static inline RowMatrix<uint8_t> create_bolt_codes(int64_t nrows,
                                                   ColMatrix<float> centroids)
{
    auto X = create_X_matrix(nrows);
    RowMatrix<uint8_t> X_enc(nrows, _M);
    bolt_encode<_M>(X.data(), nrows, total_len, centroids.data(), X_enc.data());
    return X_enc;
}

static inline void check_bolt_scan(const uint8_t* dists_u8, const uint16_t* dists_u16,
    const uint16_t* dists_u16_safe, const ColMatrix<uint8_t>& luts,
    const RowMatrix<uint8_t>& codes, int M, int64_t nblocks)
{
    for (int b = 0; b < nblocks; b++) {
        auto dist_ptr_u8 = dists_u8 + b * 32;
        auto dist_ptr_u16 = dists_u16 + b * 32;
        auto dist_ptr_u16_safe = dists_u16_safe + b * 32;
        auto codes_ptr = codes.data() + b * M * 32;
        for (int i = 0; i < 32; i++) {
            int dist_u8 = dist_ptr_u8[i];
            int dist_u16 = dist_ptr_u16[i];
            int dist_u16_safe = dist_ptr_u16_safe[i];

            // compute dist the scan should have returned based on the LUT
            int dist_true_u8 = 0;
            int dist_true_u16 = 0;
            int dist_true_u16_safe = 0;
            for (int m = 0; m < M; m++) {
                uint8_t code = codes_ptr[i + 32 * m];
                // uint8_t code = codes_ptr[i * m + 32];
                uint8_t low_bits = code & 0x0F;
                uint8_t high_bits = (code >> 4) & 0x0F;

                auto d0 = luts(low_bits, 2 * m);
                auto d1 = luts(high_bits, 2 * m + 1);

                // uint8 distances
                dist_true_u8 += d0 + d1;

                // uint16 distances
                auto pair_dist = d0 + d1;
                dist_true_u16 += (pair_dist > 255 ? 255 : pair_dist);

                // uint16 safe distance
                dist_true_u16_safe += d0 + d1;
            }
            dist_true_u8 = dist_true_u8 > 255 ? 255 : dist_true_u8;
            CAPTURE(b);
            CAPTURE(i);
            REQUIRE(dist_true_u8 == dist_u8);
            REQUIRE(dist_true_u16 == dist_u16);
            REQUIRE(dist_true_u16_safe == dist_u16_safe);
        }
    }
}
//
//static inline void print_dist_stats(const std::string& name, int64_t N,
//    double t_ms)
//{
//    printf("%s: %.2f (%.1fM/s)\n", name.c_str(), t_ms, N / (1e3 * t_ms));
//}
//
//template<class dist_t>
//static inline void print_dist_stats(const std::string& name,
//    const dist_t* dists, int64_t N, double t_ms)
//{
//    if (dists != nullptr) {
//        // prevent_optimizing_away_dists(dists, N);
//        // if (N < 100) {
//        //     auto printable_ar = ar::add(dists, N, 0);
//        //     ar::print(printable_ar.get(), N);
//        // }
//    }
//    print_dist_stats(name, N, t_ms);
//}

} // anonymous namespace
