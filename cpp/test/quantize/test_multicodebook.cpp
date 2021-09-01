

#ifdef BLAZE
    #include "test/external/catch.hpp"
    #include "src/quantize/multi_codebook.hpp"
    #include "src/external/eigen/Eigen/Dense"
    #include "src/utils/eigen_utils.hpp"
    #include "src/utils/debug_utils.hpp"
    #include "src/utils/bit_ops.hpp"
    #include "src/utils/memory.hpp"
    #include "test/testing_utils/testing_utils.hpp"
#else
    #include "catch.hpp"
    #include "multi_codebook.hpp"
    #include "Dense"
    #include "eigen_utils.hpp"
    #include "debug_utils.hpp"
    #include "bit_ops.hpp"
    #include "memory.hpp"
    #include "testing_utils.hpp"
#endif


TEST_CASE("popcnt", "[mcq][popcount]") {
    
//    std::cout << "testing mcq algorithms for popcount...\n";
    
    static constexpr int nblocks = 3;
    static constexpr int N = 32 * nblocks;
    static constexpr int M = 8;  // must be 8 for tests that cast to uint64_t

    // TODO aligned alloc if we test vectorized version
    // uint8_t X_[N * M];
    uint8_t* codes = aligned_alloc<uint8_t>(N * M);

    // set lower 4 bits in each byte to 2,4,...,2M, upper to i
    for (int i = 0; i < N; i++) {
        for (uint8_t j = 0; j < M; j++) {
            uint8_t upper_bits = (i % 16) << 4;
            codes[M * i + j] = 2 * j + upper_bits;
        }
    }
    // uint8_t* codes = &X_[0];

    // uint8_t q_[M];
    uint8_t* q = aligned_alloc<uint8_t>(M);
    for (uint8_t i = 0; i < M; i++) { // successive 4 bits are 0,1,2,...
        // q_[i] = (2 * i) + (((2 * i) + 1) << 4);
        uint8_t upper_bits = (((2 * i) + 1) % 16) << 4;
        q[i] = (2 * i) + upper_bits;
    }
    // uint8_t* q = &q_[0];

//    std::cout << "q:\n";
//    uint64_t q_uint = *(uint64_t*)q;
//    dumpEndianBits(q_uint);
    const uint64_t* q64 = reinterpret_cast<const uint64_t*>(q);

    // compute distances using our function
    uint8_t dists[N];
    // dist::popcount_8B(codes, q_uint, &dists[0], N);
    dist::popcount_generic<M>(codes, q, &dists[0], N);

    // compute distances by casting to int64 arrays
    // std::cout << "bit diffs:\n";
    for (int i = 0; i < N; i++) {
        int count = 0;
        for (int b = 0; b < M; b += 8) {
            uint64_t x = *(uint64_t*)(codes + M * i + b);
            auto q_uint = *(q64 + b);
            auto diffs = x ^ q_uint;
            count += popcount(diffs);
        }
        REQUIRE(count == dists[i]);
    }


    SECTION("8B codes, non-vectorized 4b lookup table") {
        // 1) define popcnt LUTs
        // 2) ensure that lookups with it are same as popcnt
        // 3) profile both
        // 4) variant of lut function that just uses one LUT, cuz that's all it needs

        static const uint8_t mask_low4b = 0x0F;

        // tile this so that we instead have a collection of luts
        uint8_t* popcount_luts16 = aligned_alloc<uint8_t>(M * 32);
        uint8_t* popcount_luts32 = aligned_alloc<uint8_t>(M * 2 * 32);

        for (uint8_t j = 0; j < M; j++) {
            uint8_t byte = q[j];
            uint8_t low_bits = byte & mask_low4b;
            uint8_t high_bits = (byte >> 4) & mask_low4b;

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
        }

        RowVector<uint8_t> _dists_lut(N);
        auto dists_lut = _dists_lut.data();
        dist::lut_dists_4b<M>(codes, popcount_luts16, dists_lut, N);

        for (int i = 0; i < N; i++) {
            int d_lut = dists_lut[i];
            int d = dists[i];
            REQUIRE(d == d_lut);
        }

        aligned_free(popcount_luts16);
        aligned_free(popcount_luts32);
    }

    SECTION("8B codes, vectorized lookup table") {
        REQUIRE(true);

        static const uint8_t mask_low4b = 0x0F;
        static const uint8_t block_sz_rows = 32;

        int nblocks = N / block_sz_rows;
        assert(N % block_sz_rows == 0);

        uint8_t* block_codes = aligned_alloc<uint8_t>(N * M);

        // copy row-major codes to col-major in blocks of 32
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

        // try having 32B (unpacked) luts and 16B luts that need unpacking
        uint8_t* popcount_luts16 = aligned_alloc<uint8_t>(M * block_sz_rows);
        uint8_t* popcount_luts32 = aligned_alloc<uint8_t>(M * 2 * block_sz_rows);

        // create 32B luts for vectorized lookups
        // EDIT: and 16B luts for vectorized lookups that unpack luts
        REQUIRE(block_sz_rows == 32); // following loop assumes this is true
        for (uint8_t j = 0; j < M; j++) {
            uint8_t byte = q[j];
            uint8_t low_bits = byte & mask_low4b;
            uint8_t high_bits = byte >> 4;

            // 16B lut
            auto lut_ptr = popcount_luts16 + block_sz_rows * j;
            for (uint8_t i = 0; i < 16; i++) {
                lut_ptr[i +  0] = popcount(i ^ low_bits);
                lut_ptr[i + 16] = popcount(i ^ high_bits);
            }

            // 32B lut
            lut_ptr = popcount_luts32 + block_sz_rows * 2 * j;
            for (uint8_t i = 0; i < 16; i++) {
                lut_ptr[i +  0] = popcount(i ^ low_bits);
                lut_ptr[i + 16] = popcount(i ^ low_bits);
                lut_ptr[i + 32] = popcount(i ^ high_bits);
                lut_ptr[i + 48] = popcount(i ^ high_bits);
            }
        }

        uint8_t* dists_vect = aligned_alloc<uint8_t>(N);

        // check whether we got the dists right using a naive impl
        dist::debug_lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_vect, nblocks);
        for (int i = 0; i < N; i++) {
            int d_vect = dists_vect[i];
            int d = dists[i];
            REQUIRE(d == d_vect);
        }

        // check whether we got the dists right using vectorized impl
        dist::lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_vect, nblocks);
        for (int i = 0; i < N; i++) {
            int d_vect = dists_vect[i];
            int d = dists[i];
            REQUIRE(d == d_vect);
        }

        // check whether we got the dists right using unpacking vectorized impl
        dist::lut_dists_block32_4b_unpack<M>(block_codes, popcount_luts16, dists_vect, nblocks);
        for (int i = 0; i < N; i++) {
            int d_vect = dists_vect[i];
            int d = dists[i];
            REQUIRE(d == d_vect);
        }
        
        aligned_free<uint8_t>(popcount_luts16);
        aligned_free<uint8_t>(popcount_luts32);
        aligned_free<uint8_t>(block_codes);
        aligned_free<uint8_t>(dists_vect);
    }

    aligned_free<uint8_t>(codes);
    aligned_free<uint8_t>(q);
}




