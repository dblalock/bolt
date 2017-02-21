
#include "catch.hpp"
#include "multi_codebook.hpp"

#include "timing_utils.hpp"
// #include "bit_ops.hpp"
#include "memory.hpp"

#include "testing_utils.hpp"


#define PROFILE_4bit
// #define PROFILE_8bit
// #define PROFILE_12bit
// #define PROFILE_16bit
#define PROFILE_8bit_sum_codes
#define PROFILE_8bit_raw
// #define PROFILE_float_raw

static constexpr int kNreps = 10;
// static constexpr int kNreps = 1;
static constexpr int kNtrials = 5;

static constexpr int M = 8;
// static constexpr int M = 16;
// static constexpr int M = 32;

// ------------------------ popcount function

template<class T, int N>
struct _popcount;

template<class T>
struct _popcount<T, 1> {
    static int8_t count(T x) { return __builtin_popcount((uint8_t)x); }
};
template<class T>
struct _popcount<T, 2> {
    static int8_t count(T x) { return __builtin_popcount((uint16_t)x); }
};
template<class T>
struct _popcount<T, 4> {
    // static int8_t count(T x) { std::cout << "popcount 4B\n"; return __builtin_popcountl((uint32_t)x); }
    static int8_t count(T x) { return __builtin_popcountl((uint32_t)x); }
};
template<class T>
struct _popcount<T, 8> {
    // static int8_t count(T x) { std::cout << "popcount 8B\n"; return __builtin_popcountll((uint64_t)x); }
    static int8_t count(T x) { return __builtin_popcountll((uint64_t)x); }
};

template<class T>
int8_t popcount(T x) {
    return _popcount<T, sizeof(T)>::count(x);
}

// TODO split this into smaller functions and also call from other test file
TEST_CASE("popcnt_timing", "[mcq][profile][popcount]") {
//    int nblocks = 5 * 1000 * 1000;
    // static constexpr int nblocks = 1000 * 1000;
    static constexpr int nblocks = 100 * 1000;
    int64_t N = 32 * nblocks;
    auto N_millions = static_cast<double>(N) / 1e6;
    printf("searching through %.1f million vecs (%.3fMB)...\n", N_millions, N_millions * M);

    static const uint8_t mask_low4b = 0x0F;
    static const uint8_t block_sz_rows = 32;

    // random database of codes
//    RowVector<uint8_t> codes_rowmajor_(N * M);
//    codes_rowmajor_.setRandom();
//    uint8_t* codes = codes_rowmajor_.data();
    uint8_t* codes = aligned_random_ints<uint8_t>(N * M);
    REQUIRE(codes != nullptr); // not enough memory
//    randint_inplace(codes, N * M);

    // random query
//    RowVector<uint8_t> q_(M);
//    q_.setRandom();
//    uint8_t* q = q_.data();
    uint8_t* q = aligned_random_ints<uint8_t>(M);
    REQUIRE(codes != nullptr); // not enough memory
//    randint_inplace(q, M);

    // create 32B LUTs (two copies, pre-unpacked) and 16B LUTs; we also create
    // a 256-element LUT for 8bit codes and non-vectorized search
    REQUIRE(block_sz_rows == 32);
    uint8_t* popcount_luts16 = aligned_alloc<uint8_t>(M * block_sz_rows);
    uint8_t* popcount_luts32 = aligned_alloc<uint8_t>(M * 2 * block_sz_rows);

    uint8_t* popcount_luts256b = aligned_alloc<uint8_t>(M * 256);
    uint16_t* popcount_luts256s = aligned_alloc<uint16_t>(M * 256);
    uint32_t* popcount_luts256i = aligned_alloc<uint32_t>(M * 256);
    float* popcount_luts256f = aligned_alloc<float>(M * 256);

//    uint8_t low_counts[16];
//    uint8_t high_counts[16];
    for (uint8_t j = 0; j < M; j++) {
        uint8_t byte = q[j];
        uint8_t low_bits = byte & mask_low4b;
        uint8_t high_bits = (byte >> 4) & mask_low4b;

//        uint8_t low_count = popcount(i ^ low_bits);
//        uint8_t high_count = popcount(i ^ low_bits);
//        for (uint8_t i = 0; i < 16; i++) {
//            low_counts[i]
//        }

        // just a sequence of 16B LUTs
        auto lut_ptr16 = popcount_luts16 + 16 * 2 * j;
        for (uint8_t i = 0; i < 16; i++) {
            lut_ptr16[i +  0] = popcount(i ^ low_bits);
            lut_ptr16[i + 16] = popcount(i ^ high_bits);
        }

        // two consecutive copies of each LUT, to fill 32B
        auto lut_ptr32 = popcount_luts32 + 32 * 2 * j;
        for (uint8_t i = 0; i < 16; i++) {
            lut_ptr32[i +  0] = popcount(i ^ low_bits);
            lut_ptr32[i + 16] = popcount(i ^ low_bits);
            lut_ptr32[i + 32] = popcount(i ^ high_bits);
            lut_ptr32[i + 48] = popcount(i ^ high_bits);
        }

        // 256bit LUTs (of various types) for 8bit codes
        auto lut_ptr_b = popcount_luts256b + 256 * j;
        auto lut_ptr_s = popcount_luts256s + 256 * j;
        auto lut_ptr_i = popcount_luts256i + 256 * j;
        auto lut_ptr_f = popcount_luts256f + 256 * j;
        for (uint16_t i = 0; i < 256; i++) {
            uint8_t count = popcount(static_cast<uint8_t>(i) ^ byte);
            lut_ptr_b[i] = count;
            lut_ptr_s[i] = count;
            lut_ptr_i[i] = count;
            lut_ptr_f[i] = count;
        }
    }

    // create block vertical layout version of codes
    uint8_t* block_codes = aligned_alloc<uint8_t>(N * M);
    REQUIRE(block_codes != nullptr); // not enough memory
    for (int nn = 0; nn < nblocks; nn++) { // for each block
        auto block_start_idx = nn * block_sz_rows * M;
        auto in_block_ptr = codes + block_start_idx;
        auto out_block_ptr = block_codes + block_start_idx;
        for (int i = 0; i < block_sz_rows; i++) {  // for each row
            for (int j = 0; j < M; j++) {           // for each col
                auto in_ptr = in_block_ptr + (i * M) + j;
                auto out_ptr = out_block_ptr + (j * block_sz_rows) + i;
                *out_ptr = *in_ptr;
            }
        }
    }

    // store distances from each method; first two don't actually need align,
    // but as well give all of them identical treatment
    uint8_t* dists_popcnt = aligned_alloc<uint8_t>(N);
    uint8_t* dists_scalar = aligned_alloc<uint8_t>(N);
    uint8_t* dists_vector = aligned_alloc<uint8_t>(N);
    uint8_t* dists_unpack = aligned_alloc<uint8_t>(N);
    uint8_t* dists_incorrect = aligned_alloc<uint8_t>(N);
    uint8_t* dists_incorrect2 = aligned_alloc<uint8_t>(N);
    uint8_t* dists_vertical32 = aligned_alloc<uint8_t>(N);
    uint8_t* dists_vertical64 = aligned_alloc<uint8_t>(N);
    uint8_t* dists_vertical128 = aligned_alloc<uint8_t>(N);
    uint8_t* dists_vertical256 = aligned_alloc<uint8_t>(N);

    // ================================================================ timing

//    double t_popcnt = 0, t_scalar = 0, t_vector = 0, t_vector_broken = 0;
    double t = 0;

    std::cout << "starting searches...\n";

    // ------------------------------------------------ 4bit codes
#ifdef PROFILE_4bit
    # pragma mark 4bit codes

    std::cout << "-------- dists with " << M << "B of 4bit codes\n";


    PROFILE_DIST_COMPUTATION("built-in popcnt", 5, dists_popcnt, N,
        dist::popcount_generic<M>(codes, q, dists_popcnt, N));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "built-in popcnt", kNtrials,
        dists_popcnt, N,
        dist::popcount_generic<M>(codes, q, dists_popcnt, N));


//     { // compute distances using popcnt instruction
// //        uint64_t q_uint = *(uint64_t*)q;
//         EasyTimer _(t);
//         dist::popcount_generic<M>(codes, q, dists_popcnt, N);
//     }
//     prevent_optimizing_away_dists(dists_popcnt, N);
//     printf("t_popcnt: %.2f (%.1fM/s)\n", t, N_millions / (t/1e3));


    // scalar version takes 10x longer than vectorized version
    PROFILE_DIST_COMPUTATION("naive 4b lut", 5, dists_scalar, N,
        dist::debug_lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_scalar, nblocks));
    // { // compute distances using one lookup at a time, vertical mem layout
    //     EasyTimer _(t);
    //     dist::debug_lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_scalar, nblocks);
    // }
    // prevent_optimizing_away_dists(dists_scalar, N);
    // printf("t_scalar: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("vectorized 4b lut", 5, dists_vector, N,
        dist::lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_vector, nblocks));
    // { // compute vectorized distances
    //     EasyTimer _(t);
    //     dist::lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_vector, nblocks);
    // }
    // prevent_optimizing_away_dists(dists_vector, N);
    // printf("t_vector: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("vect unpack 4b lut", 5, dists_unpack, N,
        dist::lut_dists_block32_4b_unpack<M>(block_codes, popcount_luts16, dists_unpack, nblocks));
    // { // compute vectorized distances with impl that unpacks 16B luts into 32B
    //     EasyTimer _(t);
    //     dist::lut_dists_block32_4b_unpack<M>(block_codes, popcount_luts16, dists_unpack, nblocks);
    // }
    // prevent_optimizing_away_dists(dists_unpack, N);
    // printf("t_unpack: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));


    #pragma mark four_bit vertical
    // note that we aren't passing in correct codes to these, except the first


    PROFILE_DIST_COMPUTATION("32 vertical", 5, dists_vertical32, N,
        (dist::lut_dists_block32_4b_vertical<32, M>(block_codes, popcount_luts16, dists_vertical32, N)));
    // // { // compute vectorized distances with experimental (incorrect) impl
    // //     EasyTimer _(t);
    // //     dist::lut_dists_block32_4b_vertical<32, M>(block_codes, popcount_luts16, dists_vertical32, N);
    // // }
    // // prevent_optimizing_away_dists(dists_vertical32, N);
    // // printf("t 32 vertical: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("64 vertical", 5, dists_vertical64, N,
        (dist::lut_dists_block32_4b_vertical<64, M>(block_codes, popcount_luts16, dists_vertical64, N)));
    // // { // compute vectorized distances with experimental (incorrect) impl
    // //     EasyTimer _(t);
    // //     dist::lut_dists_block32_4b_vertical<64, M>(block_codes, popcount_luts16, dists_vertical64, N);
    // // }
    // // prevent_optimizing_away_dists(dists_vertical64, N);
    // // printf("t 64 vertical: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("128 vertical", 5, dists_vertical128, N,
        (dist::lut_dists_block32_4b_vertical<128, M>(block_codes, popcount_luts16, dists_vertical128, N)));
    // // { // compute vectorized distances with experimental (incorrect) impl
    // //     EasyTimer _(t);
    // //     dist::lut_dists_block32_4b_vertical<128, M>(block_codes, popcount_luts16, dists_vertical128, N);
    // // }
    // // prevent_optimizing_away_dists(dists_vertical128, N);
    // // printf("t 128 vertical: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("256 vertical", 5, dists_vertical256, N,
        (dist::lut_dists_block32_4b_vertical<256, M>(block_codes, popcount_luts16, dists_vertical256, N)));
    // // { // compute vectorized distances with experimental (incorrect) impl
    // //     EasyTimer _(t);
    // //     dist::lut_dists_block32_4b_vertical<256, M>(block_codes, popcount_luts16, dists_vertical256, N);
    // // }
    // // prevent_optimizing_away_dists(dists_vertical256, N);
    // // printf("t 256 vertical: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

#endif // PROFILE_4bit

    aligned_free(dists_vertical64);
    aligned_free(dists_vertical128);
    aligned_free(dists_vertical256);


    // ------------------------------------------------ 8bit codes
    # pragma mark 8bit codes

#ifdef PROFILE_8bit
    std::cout << "-------- dists with " << M << "B of 16bit codes\n";

    uint8_t*  dists_8b_b = aligned_alloc<uint8_t>(N);
    uint16_t* dists_8b_s = aligned_alloc<uint16_t>(N);
    uint32_t* dists_8b_i = aligned_alloc<uint32_t>(N);
    float*    dists_8b_f = aligned_alloc<float>(N);

    PROFILE_DIST_COMPUTATION("t 8b, 8b dist", 5, dists_8b_b, N,
        dist::lut_dists_8b<M>(codes, popcount_luts256b, dists_8b_b, N));
    // { // 8bit lookups, 8bit distances
    //     EasyTimer _(t);
    //     dist::lut_dists_8b<M>(codes, popcount_luts256b, dists_8b_b, N);
    // }
    // prevent_optimizing_away_dists(dists_8b_b, N);
    // printf("t 8b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 8b, 16b dist", 5, dists_8b_s, N,
        dist::lut_dists_8b<M>(codes, popcount_luts256s, dists_8b_s, N));
    // { // 8bit lookups, 16bit distances
    //     EasyTimer _(t);
    //     dist::lut_dists_8b<M>(codes, popcount_luts256s, dists_8b_s, N);
    // }
    // prevent_optimizing_away_dists(dists_8b_s, N);
    // printf("t 8b, 16b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 8b, 32b dist", 5, dists_8b_i, N,
        dist::lut_dists_8b<M>(codes, popcount_luts256i, dists_8b_i, N));
    // { // 8bit lookups, 32bit distances
    //     EasyTimer _(t);
    //     dist::lut_dists_8b<M>(codes, popcount_luts256i, dists_8b_i, N);
    // }
    // prevent_optimizing_away_dists(dists_8b_i, N);
    // printf("t 8b, 32b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 8b, float dist", 5, dists_8b_f, N,
        dist::lut_dists_8b<M>(codes, popcount_luts256f, dists_8b_f, N));
    // { // 8bit lookups, 32bit float distances
    //     EasyTimer _(t);
    //     dist::lut_dists_8b<M>(codes, popcount_luts256f, dists_8b_f, N);
    // }
    // prevent_optimizing_away_dists(dists_8b_f, N);
    // printf("t 8b, float dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    #pragma mark 8bit vertical

    PROFILE_DIST_COMPUTATION("t 32 8b, 8b dist", 5, dists_8b_b, N,
        (dist::lut_dists_8b_vertical<32, M>(block_codes, popcount_luts256b, dists_8b_b, N)));
    // { // 8bit lookups
    //  EasyTimer _(t);
    //  dist::lut_dists_8b_vertical<32, M>(block_codes, popcount_luts256b, dists_8b_b, N);
    // }
    // prevent_optimizing_away_dists(dists_8b_b, N);
    // printf("t 32 8b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

#endif // PROFILE_8bit

    // ------------------------------------------------ <NOTE>
    // below this point, no effort made at achieving correctness;
    // scans are operating on unitialized memory
    // ------------------------------------------------ </NOTE>

    // ------------------------------------------------ 12bit codes
    # pragma mark 12bit codes

#ifdef PROFILE_12bit
    std::cout << "-------- dists with " << M << "B of 12bit codes\n";

    uint8_t*  dists_12b_b = aligned_alloc<uint8_t>(N);
    uint16_t*  dists_12b_s = aligned_alloc<uint16_t>(N);
    uint32_t*  dists_12b_i = aligned_alloc<uint32_t>(N);

    int lut_sz_12b = (1 << 12);
    uint8_t* popcount_luts12b_b = aligned_random_ints<uint8_t>(M * lut_sz_12b);
    uint16_t* popcount_luts12b_s = aligned_random_ints<uint16_t>(M * lut_sz_12b);
    uint32_t* popcount_luts12b_i = aligned_random_ints<uint32_t>(M * lut_sz_12b);

    PROFILE_DIST_COMPUTATION("t 12b, 8b dist", 5, dists_12b_b, N,
        dist::lut_dists_12b<M>(codes, popcount_luts12b_b, dists_12b_b, N));
//     // { // 12bit lookups, 8bit distances
//     //     EasyTimer _(t);
//     //     dist::lut_dists_12b<M>(codes, popcount_luts12b_b, dists_12b_b, N);
//     // }
//     // prevent_optimizing_away_dists(dists_12b_b, N);
//     // printf("t 12b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 12b, 16b dist", 5, dists_12b_s, N,
        dist::lut_dists_12b<M>(codes, popcount_luts12b_s, dists_12b_s, N));
//     { // 12bit lookups, 16bit distances
//         EasyTimer _(t);
//         dist::lut_dists_12b<M>(codes, popcount_luts12b_s, dists_12b_s, N);
//     }
//     prevent_optimizing_away_dists(dists_12b_b, N);
//     printf("t 12b, 16b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 12b, 16b dist", 5, dists_12b_i, N,
        dist::lut_dists_12b<M>(codes, popcount_luts12b_i, dists_12b_i, N));
//     { // 12bit lookups, 32bit distances
//         EasyTimer _(t);
//         dist::lut_dists_12b<M>(codes, popcount_luts12b_i, dists_12b_i, N);
//     }
//     prevent_optimizing_away_dists(dists_12b_b, N);
//     printf("t 12b, 32b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    #pragma mark 12bit vertical

    PROFILE_DIST_COMPUTATION("t 16 12b, 8b dist", 5, dists_12b_b, N,
        (dist::lut_dists_12b_vertical<16, M>(block_codes, popcount_luts12b_b, dists_12b_b, N)));
//     { // 12bit lookups, 8bit distances
//         EasyTimer _(t);
//         dist::lut_dists_12b_vertical<16, M>(block_codes, popcount_luts12b_b, dists_12b_b, N);
//     }
//     prevent_optimizing_away_dists(dists_12b_b, N);
//     printf("t 16, 12b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 64 12b, 8b dist", 5, dists_12b_b, N,
        (dist::lut_dists_12b_vertical<64, M>(block_codes, popcount_luts12b_b, dists_12b_b, N)));
//     { // 12bit lookups, 8bit distances
//         EasyTimer _(t);
//         dist::lut_dists_12b_vertical<64, M>(block_codes, popcount_luts12b_b, dists_12b_b, N);
//     }
//     prevent_optimizing_away_dists(dists_12b_b, N);
//     printf("t 64, 12b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 256 12b, 8b dist", 5, dists_12b_b, N,
        (dist::lut_dists_12b_vertical<256, M>(block_codes, popcount_luts12b_b, dists_12b_b, N)));
//     { // 12bit lookups, 8bit distances
//         EasyTimer _(t);
//         dist::lut_dists_12b_vertical<256, M>(block_codes, popcount_luts12b_b, dists_12b_b, N);
//     }
//     prevent_optimizing_away_dists(dists_12b_b, N);
//     printf("t 256, 12b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

#endif // PROFILE_12bit

    // ------------------------------------------------ 16bit codes
    # pragma mark 16bit codes

#ifdef PROFILE_16bit
    std::cout << "-------- dists with " << M << "B of 16bit codes\n";

    uint8_t*  dists_16b_b = aligned_alloc<uint8_t>(N);
    uint16_t*  dists_16b_s = aligned_alloc<uint16_t>(N);
    uint32_t*  dists_16b_i = aligned_alloc<uint32_t>(N);

    int lut_sz_16b = (1 << 16);
    uint8_t* popcount_luts16b_b = aligned_random_ints<uint8_t>(M/2 * lut_sz_16b);
    uint16_t* popcount_luts16b_s = aligned_random_ints<uint16_t>(M/2 * lut_sz_16b);
    uint32_t* popcount_luts16b_i = aligned_random_ints<uint32_t>(M/2 * lut_sz_16b);

    uint16_t* codes16 = (uint16_t*)codes;
    uint16_t* block_codes16 = (uint16_t*)block_codes;

    PROFILE_DIST_COMPUTATION("t 16b, 8b dist", 5, dists_16b_b, N,
        dist::lut_dists_16b<M>(codes16, popcount_luts16b_b, dists_16b_b, N));
//     { // 16bit lookups, 8bit distances
//         EasyTimer _(t);
//         dist::lut_dists_16b<M>(codes16, popcount_luts16b_b, dists_16b_b, N);
//     }
//     prevent_optimizing_away_dists(dists_16b_b, N);
//     printf("t 16b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 16b, 16b dist", 5, dists_16b_s, N,
        dist::lut_dists_16b<M>(codes16, popcount_luts16b_s, dists_16b_s, N));
//     { // 16bit lookups, 16bit distances
//         EasyTimer _(t);
//         dist::lut_dists_16b<M>(codes16, popcount_luts16b_s, dists_16b_s, N);
//     }
//     prevent_optimizing_away_dists(dists_16b_s, N);
//     printf("t 16b, 16b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 16b, 32b dist", 5, dists_16b_i, N,
        dist::lut_dists_16b<M>(codes16, popcount_luts16b_i, dists_16b_i, N));
//     { // 16bit lookups, 16bit distances
//         EasyTimer _(t);
//         dist::lut_dists_16b<M>(codes16, popcount_luts16b_i, dists_16b_i, N);
//     }
//     prevent_optimizing_away_dists(dists_16b_i, N);
//     printf("t 16b, 32b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));


    #pragma mark 16bit vertical

    PROFILE_DIST_COMPUTATION("t 32 16b, 8b dist", 5, dists_16b_b, N,
        (dist::lut_dists_16b_vertical<32, M>(block_codes16, popcount_luts16b_b, dists_16b_b, N)));
//     { // 16bit lookups, 8bit distances
//         EasyTimer _(t);
//         dist::lut_dists_16b_vertical<32, M>(block_codes16, popcount_luts16b_b, dists_16b_b, N);
//     }
//     prevent_optimizing_away_dists(dists_16b_b, N);
//     printf("t 32 16b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 256 16b, 8b dist", 5, dists_16b_b, N,
        (dist::lut_dists_16b_vertical<256, M>(block_codes16, popcount_luts16b_b, dists_16b_b, N)));
//     { // 16bit lookups, 8bit distances
//         EasyTimer _(t);
//         dist::lut_dists_16b_vertical<256, M>(block_codes16, popcount_luts16b_b, dists_16b_b, N);
//     }
//     prevent_optimizing_away_dists(dists_16b_b, N);
//     printf("t 256 16b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 1024 16b, 8b dist", 5, dists_16b_b, N,
        (dist::lut_dists_16b_vertical<1024, M>(block_codes16, popcount_luts16b_b, dists_16b_b, N)));
//     { // 16bit lookups, 8bit distances
//         EasyTimer _(t);
//         dist::lut_dists_16b_vertical<1024, M>(block_codes16, popcount_luts16b_b, dists_16b_b, N);
//     }
//     prevent_optimizing_away_dists(dists_16b_b, N);
//     printf("t 1024 16b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

#endif // PROFILE_16bit

    // ------------------------------------------------ floats
    # pragma mark floats

#ifdef PROFILE_float_raw
    static constexpr uint64_t nblocksf = 10000;
    static constexpr uint64_t Nf = 256 * nblocksf;
    static constexpr double Nf_millions = static_cast<double>(Nf) / 1e6;
    static constexpr int Df = 16;
    float* Xf = aligned_random<float>(Nf * Df);
    float* qf = aligned_random<float>(Df);
    float* distsf = aligned_alloc<float>(Nf);

    std::cout << "-------- float distances where D = " << Df << "\n";

    PROFILE_DIST_COMPUTATION("t 8 floats full dist", 5, distsf, Nf,
        (dist::float_dists_vertical<8, Df>(Xf, qf, distsf, Nf)));
//     { // operate on 8 floats per col (1 stripes)
//         EasyTimer _(t);
//         dist::float_dists_vertical<8, Df>(Xf, qf, distsf, Nf);
//     }
//     prevent_optimizing_away_dists(distsf, Nf);
//     printf("t 8 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 16 floats full dist", 5, distsf, Nf,
        (dist::float_dists_vertical<16, Df>(Xf, qf, distsf, Nf)));
//     { // operate on 16 floats per col (2 stripes)
//         EasyTimer _(t);
//         dist::float_dists_vertical<16, Df>(Xf, qf, distsf, Nf);
//     }
//     prevent_optimizing_away_dists(distsf, Nf);
//     printf("t 16 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 32 floats full dist", 5, distsf, Nf,
        (dist::float_dists_vertical<32, Df>(Xf, qf, distsf, Nf)));
//     { // operate on 32 floats per col (4 stripes)
//         EasyTimer _(t);
//         dist::float_dists_vertical<32, Df>(Xf, qf, distsf, Nf);
//     }
//     prevent_optimizing_away_dists(distsf, Nf);
//     printf("t 32 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 128 floats full dist", 5, distsf, Nf,
        (dist::float_dists_vertical<128, Df>(Xf, qf, distsf, Nf)));
//     { // operate on 128 floats per col (16 stripes)
//         EasyTimer _(t);
//         dist::float_dists_vertical<128, Df>(Xf, qf, distsf, Nf);
//     }
//     prevent_optimizing_away_dists(distsf, Nf);
//     printf("t 128 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 256 floats full dist", 5, distsf, Nf,
        (dist::float_dists_vertical<256, Df>(Xf, qf, distsf, Nf)));
//     { // operate on 256 floats per col (32 stripes)
//         EasyTimer _(t);
//         dist::float_dists_vertical<256, Df>(Xf, qf, distsf, Nf);
//     }
//     prevent_optimizing_away_dists(distsf, Nf);
//     printf("t 256 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 512 floats full dist", 5, distsf, Nf,
        (dist::float_dists_vertical<512, Df>(Xf, qf, distsf, Nf)));
//     { // operate on 512 floats per col (32 stripes)
//         EasyTimer _(t);
//         dist::float_dists_vertical<512, Df>(Xf, qf, distsf, Nf);
//     }
//     prevent_optimizing_away_dists(distsf, Nf);
//     printf("t 512 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 1024 floats full dist", 5, distsf, Nf,
        (dist::float_dists_vertical<1024, Df>(Xf, qf, distsf, Nf)));
//     { // operate on 1024 floats per col (32 stripes)
//         EasyTimer _(t);
//         dist::float_dists_vertical<1024, Df>(Xf, qf, distsf, Nf);
//     }
//     prevent_optimizing_away_dists(distsf, Nf);
//     printf("t 1024 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));


    aligned_free(Xf);
    aligned_free(qf);
    aligned_free(distsf);

#endif // PROFILE_float_raw

    // ------------------------------------------------ int8s
    # pragma mark int8s

#ifdef PROFILE_8bit_sum_codes
    static constexpr uint64_t nblocksb = 10000;
    static constexpr uint64_t Nb = 256 * nblocksb;
    static constexpr double Nb_millions = static_cast<double>(Nb) / 1e6;
    static constexpr int Db = 16;
    uint8_t* Xb = aligned_random_ints<uint8_t>(Nb * Db);
    uint8_t* qb = aligned_random_ints<uint8_t>(Db);
    uint16_t* distsb = aligned_alloc<uint16_t>(Nb);
    uint8_t* distsb8 = aligned_alloc<uint8_t>(Nb);
//    uint16_t* distsb2 = aligned_alloc<uint16_t>(Nb);

    std::cout << "-------- int8 code sums where D = " << Db << "\n";

    PROFILE_DIST_COMPUTATION("t 32 int8s just sums", 5, distsb8, Nb,
        (dist::sum_inputs<32, Db>(Xb, qb, distsb8, Nb)));
//     { // just sum inputs as a lower bound on how fast any func could be
//         EasyTimer _(t);
//         dist::sum_inputs<32, Db>(Xb, qb, distsb8, Nb);
//     }
//     prevent_optimizing_away_dists(distsb8, Nb);
//     printf("t 32 int8s just sums: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 128 int8s just sums", 5, distsb8, Nb,
        (dist::sum_inputs<128, Db>(Xb, qb, distsb8, Nb)));
//     { // just sum inputs as a lower bound on how fast any func could be
//         EasyTimer _(t);
//         dist::sum_inputs<128, Db>(Xb, qb, distsb8, Nb);
//     }
//     prevent_optimizing_away_dists(distsb8, Nb);
//     printf("t 128 int8s just sums: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 256 int8s just sums", 5, distsb8, Nb,
        (dist::sum_inputs<256, Db>(Xb, qb, distsb8, Nb)));
//     { // just sum inputs as a lower bound on how fast any func could be
//         EasyTimer _(t);
//         dist::sum_inputs<256, Db>(Xb, qb, distsb8, Nb);
//     }
//     prevent_optimizing_away_dists(distsb8, Nb);
//     printf("t 256 int8s just sums: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 512 int8s just sums", 5, distsb8, Nb,
        (dist::sum_inputs<512, Db>(Xb, qb, distsb8, Nb)));
//     { // just sum inputs as a lower bound on how fast any func could be
//         EasyTimer _(t);
//         dist::sum_inputs<512, Db>(Xb, qb, distsb8, Nb);
//     }
//     prevent_optimizing_away_dists(distsb8, Nb);
//     printf("t 512 int8s just sums: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

#endif // PROFILE_8bit_sum_codes
#ifdef PROFILE_8bit_raw
    std::cout << "-------- int8 distances where D = " << Db << "\n";

    PROFILE_DIST_COMPUTATION("t 32 int8s full dist", 5, distsb8, Nb,
        (dist::byte_dists_vertical<32, Db>(Xb, qb, distsb, Nb)));
//     { // operate on 32 int8s per col (1 stripe)
//         EasyTimer _(t);
//         dist::byte_dists_vertical<32, Db>(Xb, qb, distsb, Nb);
//     }
//     prevent_optimizing_away_dists(distsb, Nb);
//     printf("t 32 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
// //    for (int64_t i = 0; i < Nb; i++) { Xb[i] += 1; }

    PROFILE_DIST_COMPUTATION("t 64 int8s full dist", 5, distsb8, Nb,
        (dist::byte_dists_vertical<64, Db>(Xb, qb, distsb, Nb)));
//     { // operate on 64 int8s per col (2 stripes)
//         EasyTimer _(t);
//         dist::byte_dists_vertical<64, Db>(Xb, qb, distsb, Nb);
//     }
//     prevent_optimizing_away_dists(distsb, Nb);
//     printf("t 64 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
// //    for (int64_t i = 0; i < Nb; i++) { Xb[i] += 1; }

    PROFILE_DIST_COMPUTATION("t 128 int8s full dist", 5, distsb8, Nb,
        (dist::byte_dists_vertical<128, Db>(Xb, qb, distsb, Nb)));
//     { // operate on 128 int8s per col (4 stripes)
//         EasyTimer _(t);
//         dist::byte_dists_vertical<128, Db>(Xb, qb, distsb, Nb);
//     }
//     prevent_optimizing_away_dists(distsb, Nb);
//     printf("t 128 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 256 int8s full dist", 5, distsb8, Nb,
        (dist::byte_dists_vertical<256, Db>(Xb, qb, distsb, Nb)));
//     { // operate on 256 int8s per col (8 stripes)
//         EasyTimer _(t);
//         dist::byte_dists_vertical<256, Db>(Xb, qb, distsb, Nb);
//     }
//     prevent_optimizing_away_dists(distsb, Nb);
//     printf("t 256 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

    PROFILE_DIST_COMPUTATION("t 512 int8s full dist", 5, distsb8, Nb,
        (dist::byte_dists_vertical<512, Db>(Xb, qb, distsb, Nb)));
//     { // operate on 512 int8s per col (8 stripes)
//         EasyTimer _(t);
//         dist::byte_dists_vertical<512, Db>(Xb, qb, distsb, Nb);
//     }
//     prevent_optimizing_away_dists(distsb, Nb);
//     printf("t 512 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));


    aligned_free(Xb);
    aligned_free(qb);
    aligned_free(distsb);

#endif // PROFILE_8bit_raw

    // ------------------------------------------------ correctness

    bool same0 = true;
    bool same1 = true;
    bool same2 = true;
    bool same3 = true;
    bool same_8b_b = true, same_8b_s = true, same_8b_i = true, same_8b_f = true;
    for (int64_t n = 0; n < N; n++) {
#ifdef PROFILE_4bit
        same0 &= dists_popcnt[n] == dists_scalar[n];
        same1 &= dists_scalar[n] == dists_vector[n];
        same2 &= dists_vector[n] == dists_unpack[n];
        same3 &= dists_unpack[n] == dists_vertical32[n];
#endif
#ifdef PROFILE_8bit
        same_8b_b &= dists_popcnt[n] == dists_8b_b[n];
        same_8b_s &= dists_8b_b[n] == dists_8b_s[n];
        same_8b_i &= dists_8b_s[n] == dists_8b_i[n];
        same_8b_f &= dists_8b_i[n] == dists_8b_f[n];
#endif

//        if (n < 50 && !same0) {
//            printf("popcnt dist, scalar dist = %d, %d\n", dists_popcnt[n], dists_scalar[n]);
//        }
//        same1 &= dists_popcnt[n] == dists_vector[n];
//        same2 &= dists_popcnt[n] == dists_unpack[n];

//        REQUIRE(dists_popcnt[n] == dists_scalar[n]);
//        REQUIRE(dists_popcnt[n] == dists_vector[n]);
    }
#ifdef PROFILE_4bit
    REQUIRE(same0);
    REQUIRE(same1);
    REQUIRE(same2);
    REQUIRE(same3);
    std::cout << "same0? " << same0 << "\n"; // 0, so popcnt prolly broken
    std::cout << "same1? " << same1 << "\n";
    std::cout << "same2? " << same2 << "\n";
    std::cout << "same3? " << same3 << "\n";
#endif
#ifdef PROFILE_8bit
    std::cout << "same_8b_b? " << same_8b_b << "\n";
    std::cout << "same_8b_s? " << same_8b_s << "\n";
    std::cout << "same_8b_i? " << same_8b_i << "\n";
    std::cout << "same_8b_f? " << same_8b_f << "\n";
#endif
    std::cout << "done" << std::endl;

    aligned_free(q);
    aligned_free(codes);
    aligned_free(block_codes);
    aligned_free(popcount_luts16);
    aligned_free(popcount_luts32);
    aligned_free(dists_popcnt);
    aligned_free(dists_scalar);
    aligned_free(dists_vector);
    aligned_free(dists_unpack);
    aligned_free(dists_vertical32);
}
