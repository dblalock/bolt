//
//  sketch.hpp
//  Bolt
//
//  Created by DB on 12/9/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef sketch_hpp
#define sketch_hpp

#include <stdio.h>

#ifdef BLAZE
    #include "src/utils/eigen_utils.hpp"
#else
    #include "eigen_utils.hpp"
#endif


namespace {
// template<typename ColMatrixT>
// void _osnap_sketch_cols(const ColMatrixT0& A, RowMatrix<uint32_t> all_idxs,
template<bool SketchCols=true, typename MatrixT0, typename MatrixT1>
void osnap_sketch(const MatrixT0& A, RowMatrix<uint32_t> all_idxs,
                        RowMatrix<float> all_signs, MatrixT1&& out)
{
    // all_idxs is ncols x nsketches
    // printf("sketch_cols\n");
    // printf("all_idxs rows, cols = %ld, %ld\n", all_idxs.rows(), all_idxs.cols());
    // printf("A rows, cols = %ld, %ld\n", A.rows(), A.cols());
    assert(all_idxs.rows() == A.cols());
    auto s = all_idxs.cols();
    if (s > 1) {
        float scale = 1.f / sqrt(s);
        for (int i = 0; i < all_idxs.rows(); i++) {
            for (int j = 0; j < all_idxs.cols(); j++) {
                auto out_col_idx = all_idxs(i, j);
                if (SketchCols) {
                    out.col(out_col_idx) += A.col(i) * all_signs(i, j) * scale;
                } else {
                    out.row(out_col_idx) += A.row(i) * all_signs(i, j) * scale;
                }
            }
        }
    } else { // eliminate unneeded multiply from innermost loop
        for (int i = 0; i < all_idxs.rows(); i++) {
            for (int j = 0; j < all_idxs.cols(); j++) {
                auto out_col_idx = all_idxs(i, j);
                if (SketchCols) {
                    out.col(out_col_idx) += A.col(i) * all_signs(i, j);
                } else {
                    out.row(out_col_idx) += A.row(i) * all_signs(i, j);
                }
            }
        }
    }
}

// template<typename RowMatrixT0, typename RowMatrixT1>
// void _osnap_sketch_rows(const RowMatrixT0& A, RowMatrix<uint32_t> all_idxs,
//                         RowMatrix<float> all_signs, RowMatrixT1&& out)
// {
//     // all_idxs is ncols x nsketches
//     // printf("sketch_rows\n");
//     // printf("all_idxs rows, cols = %ld, %ld\n", all_idxs.rows(), all_idxs.cols());
//     // printf("A rows, cols = %ld, %ld\n", A.rows(), A.cols());
//     assert(all_idxs.rows() == A.rows());
//     assert(all_signs.rows() == A.rows());
//     auto s = all_idxs.cols();
//     float scale = 1.f / sqrt(s);
//     for (int i = 0; i < all_idxs.rows(); i++) {
//         for (int j = 0; j < all_idxs.cols(); j++) {
//             auto out_col_idx = all_idxs(i, j);
//             out.row(out_col_idx) += A.row(i) * all_signs(i, j) * scale;
//         }
//     }
// }

} // end anonymous namespace

struct OsnapSketch {

    OsnapSketch(int D, int d_total, int nsketches):
        D(D), d_total(d_total), nsketches(nsketches),
        sketch_idxs(D, nsketches), sketch_signs(D, nsketches)
    {
        assert(nsketches <= d_total);
        std::random_device rd;
        std::mt19937 rng(rd()); // twister not ideal, but cpp missing better ones?
        // std::uniform_int_distribution<uint32_t> idx_distro(0, d - 1);
        std::uniform_int_distribution<uint32_t> sign_distro(0, 1);
        auto start_idx = 0;
        auto sketch_len = d_total / nsketches;
        auto tail_sz = d_total % nsketches;
        for (int s = 0; s < nsketches; s++) {
            auto end_idx = start_idx + sketch_len;
            if (s < tail_sz) {  // handle d_total % nsketches != 0
                end_idx++;
            }
            // printf("D, d_total, s, end_idx = %d, %d, %d, %d\n", D, d_total, s, end_idx);
            assert(end_idx - 1 < d_total);
            std::uniform_int_distribution<uint32_t> idx_distro(
                start_idx, end_idx - 1);
            end_idx = start_idx;
            for (int i = 0; i < D; i++) {
                sketch_idxs(i, s) = idx_distro(rng);
                sketch_signs(i, s) = (2 * sign_distro(rng)) - 1;
            }
        }
    }

  template<typename T>
    void operator()(const ColMatrix<T>& A, ColMatrix<T>out,
                    bool transpose=false) const
    {
        if (transpose) {
            osnap_sketch<false>(
                A.transpose(), sketch_idxs, sketch_signs, out.transpose());
        } else {
            osnap_sketch(A, sketch_idxs, sketch_signs, out);
        }
    }

    int D, d_total, nsketches;
    RowMatrix<uint32_t> sketch_idxs;
    RowMatrix<float> sketch_signs;
};

// void generate_hash_sketch_idxs_and_signs(
//     int D, int d, uint32_t* idxs_out, float* signs_out)
// {
//     std::random_device rd;
//     std::mt19937 rng(rd()); // twister not ideal, but cpp missing better ones?
//     std::uniform_int_distribution<uint32_t> idx_distro(0, d - 1);
//     std::uniform_int_distribution<uint32_t> sign_distro(0, 1);
//     for int (i = 0; i < D; i++) {
//         *idxs_out++ = idx_distro(rng);
//         *signs_out++ = (2 * sign_distro(rng)) - 1;
//     }
// }

// void generate_osnap_idxs_and_signs(int D, int d_total, int nsketches,
//                                    uint32_t* idxs_out, float* signs_out)
// {
//     auto tail_sz = D % nsketches;
//     for (int s = 0; s < nsketches; s++) {
//         auto d = d_total / nsketches;
//         if (s < tail_sz) {
//             d++;
//         }
//         generate_hash_sketch_idxs_and_signs(D, d, idxs_out, signs_out);
//         idxs_out +=
//     }
// }

// template<typename T, typename OutMatrixT>


#endif /* sketch_hpp */
