
#ifdef BLAZE
	//#include "src/utils/avx_utils.hpp"
	#include "avx_utils.hpp"
#else
	#include "avx_utils.hpp"
#endif

// N has to be a multiple of 8; better if D a multiple of 3 or 4, M a multiple
// of 2 or 3
void sgemm_colmajor(const float* A, const float *B, int N, int D, int M,
                    float* out)
{
    if (N < 1 || D < 1 || M < 1) { return; }
    if (D == 1 && M <= 6) {
        switch(M) {
            case 1: sgemm_colmajor_narrow_padded<1, 1>(A, B, N, D, M, out); return;
            case 2: sgemm_colmajor_narrow_padded<1, 2>(A, B, N, D, M, out); return;
            case 3: sgemm_colmajor_narrow_padded<1, 3>(A, B, N, D, M, out); return;
            case 4: sgemm_colmajor_narrow_padded<1, 4>(A, B, N, D, M, out); return;
            case 5: sgemm_colmajor_narrow_padded<1, 5>(A, B, N, D, M, out); return;
            case 6: sgemm_colmajor_narrow_padded<1, 6>(A, B, N, D, M, out); return;
            default: return;
        }
    }
    if (D <= 4 && M <= 4) {
        switch(10 * D + M) {
            case 11: sgemm_colmajor_narrow_padded<1, 1>(A, B, N, D, M, out); return;
            case 12: sgemm_colmajor_narrow_padded<1, 2>(A, B, N, D, M, out); return;
            case 13: sgemm_colmajor_narrow_padded<1, 3>(A, B, N, D, M, out); return;
            case 14: sgemm_colmajor_narrow_padded<1, 4>(A, B, N, D, M, out); return;
            case 21: sgemm_colmajor_narrow_padded<2, 1>(A, B, N, D, M, out); return;
            case 22: sgemm_colmajor_narrow_padded<2, 2>(A, B, N, D, M, out); return;
            case 23: sgemm_colmajor_narrow_padded<2, 3>(A, B, N, D, M, out); return;
            case 24: sgemm_colmajor_narrow_padded<2, 2>(A, B, N, D, M, out); return;
            case 31: sgemm_colmajor_narrow_padded<3, 1>(A, B, N, D, M, out); return;
            case 32: sgemm_colmajor_narrow_padded<3, 2>(A, B, N, D, M, out); return;
            case 33: sgemm_colmajor_narrow_padded<3, 3>(A, B, N, D, M, out); return;
            case 34: sgemm_colmajor_narrow_padded<3, 2>(A, B, N, D, M, out); return;
            case 41: sgemm_colmajor_narrow_padded<4, 1>(A, B, N, D, M, out); return;
            case 42: sgemm_colmajor_narrow_padded<4, 2>(A, B, N, D, M, out); return;
            case 43: sgemm_colmajor_narrow_padded<4, 3>(A, B, N, D, M, out); return;
            case 44: sgemm_colmajor_narrow_padded<4, 2>(A, B, N, D, M, out); return;
            default: return;
        }
    }
    auto D_tail = D % 4;
    auto M_tail = M % 3;
    // auto D_over4 = D / 4;
    // auto M_over3 = M / 3;
    auto D_round = D - D_tail;
    auto M_round = M - M_tail;

//    auto A_row_stride = 1;
//    auto B_row_stride = 1;
//    auto out_row_stride = 1;
    auto A_col_stride = N;
    auto B_col_stride = D;
    auto out_col_stride = N;

    auto A_coltail = A + (D_round * A_col_stride);
    auto B_rowtail = B + (D_round);
    auto B_coltail = B + (M_round * B_col_stride);
    auto B_tailtail = B_coltail + D_round;
    auto out_coltail = out + (M_round * out_col_stride);

    // PRINT_VAR(out_col_stride);

    auto pos_D_round = D_round > 0;
    auto pos_M_round = M_round > 0;
    auto pos_round_mat = pos_D_round && pos_M_round;

    // PRINT_VAR(N);
    // PRINT_VAR(D);
    // PRINT_VAR(M);

    if (D >= 4 && M >= 3) {
        sgemm_colmajor_narrow_padded<4, 3>(A, B, N, D_round, M_round, out, false, A_col_stride, B_col_stride, out_col_stride);
    } else if (D % 4 == 0) {
        if (M % 2 == 0) {
            sgemm_colmajor_narrow_padded<4, 2>(A, B, N, D, M, out, false);
            return;
        } else {
            sgemm_colmajor_narrow_padded<4, 1>(A, B, N, D, M, out, false);
            return;
        }
    } else if (D % 3 == 0) {
        if (M % 2 == 0) {
            sgemm_colmajor_narrow_padded<3, 2>(A, B, N, D, M, out, false);
            return;
        } else {
            sgemm_colmajor_narrow_padded<3, 1>(A, B, N, D, M, out, false);
            return;
        }
    } else if (D % 2 == 0) {
        if (M % 2 == 0) {
            // printf("running special case; using <2, 2>\n");
            sgemm_colmajor_narrow_padded<2, 2>(A, B, N, D, M, out, false);
            return;
        } else {
            sgemm_colmajor_narrow_padded<2, 1>(A, B, N, D, M, out, false);
            return;
        }
    } else {
        // TODO break up into multiple matmuls using <4, 3> if possible
        sgemm_colmajor_narrow_padded<1, 1>(A, B, N, D, M, out, false);
        return;
    }
    // if (D < 4) { // M must be at least 5 or we would have handled this
    //     switch(D) {
    //         case 1:
    //             sgemm_colmajor_narrow_padded<1, 3>(A, B, N, D, M_round, out, false, A_col_stride, B_col_stride, out_col_stride);
    //             if (M_)
    //             break;
    //         case 2: sgemm_colmajor_narrow_padded<2, 3>(A, B, N, D, M_round, out, false, A_col_stride, B_col_stride, out_col_stride); break;
    //         case 3: sgemm_colmajor_narrow_padded<3, 3>(A, B, N, D, M_round, out, false, A_col_stride, B_col_stride, out_col_stride); break;
    //     }
    // } else if (M < 3) { // D must be at least 5 or would have handled this
    //     // switch(M) {
    //     //     case 1: sgemm_colmajor_narrow_padded<4, 1>(A, B, N, D_round, M_round, out, false, A_col_stride, B_col_stride, out_col_stride); break;
    //     //     case 2: sgemm_colmajor_narrow_padded<4, 2>(A, B, N, D_round, M_round, out, false, A_col_stride, B_col_stride, out_col_stride); break;
    //     // }
    // } else { // D >= 4 && M >= 3
    //     // do this as many times as possible; stuff below handles trailing dims
    //     sgemm_colmajor_narrow_padded<4, 3>(A, B, N, D_round, M_round, out, false, A_col_stride, B_col_stride, out_col_stride);
    // }

    // printf("case %d\n", D_tail * 10 + M_tail);
    switch (D_tail * 10 + M_tail) {
        // case 0: sgemm_colmajor_narrow_padded<4, 3>(A, B, N, D, M, out); return;
        case 0: return;
        case 10:  // one trailing input dim
            if (M % 2 == 0) {  // implies m % 6 == 0, since m % 3 == 0
                sgemm_colmajor_narrow_padded<1, 6>(
                    A_coltail, B_rowtail, N, 1, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            } else {
                sgemm_colmajor_narrow_padded<1, 3>(
                    A_coltail, B_rowtail, N, 1, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            }
            // sgemm_colmajor_narrow_padded<1, 3>(
            //         A_coltail, B_rowtail, N, 1, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 20:  // two trailing input dims
            sgemm_colmajor_narrow_padded<2, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 30: // three trailing input dims
            sgemm_colmajor_narrow_padded<3, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 1: // one trailing output dim
            sgemm_colmajor_narrow_padded<4, 1>(
                A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 2: // two trailing output dims
            sgemm_colmajor_narrow_padded<4, 2>(
                A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            return;

        // now the tricky cases: trailing input *and* output dims
        case 11: // one trailing input and one trailing output dim
            // PRINT_VAR(A_col_stride);
            // PRINT_VAR(B_col_stride);
            // PRINT_VAR(out_col_stride);
            // compute rest of the partial output
            // TODO not necessarily 3 as 2nd template arg
            sgemm_colmajor_narrow_padded<1, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            // compute remaining output col from most of A, then tail
            sgemm_colmajor_narrow_padded<4, 1>(A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<1, 1>(A_coltail, B_tailtail, N, D_tail, M_tail, out_coltail, pos_D_round, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 12: // trailing inputs, outputs: 1, 2
            sgemm_colmajor_narrow_padded<1, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<4, 2>(A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<1, 2>(A_coltail, B_tailtail, N, D_tail, M_tail, out_coltail, pos_D_round, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 21:  // trailing inputs, outputs: 2, 1
            sgemm_colmajor_narrow_padded<2, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<4, 1>(A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<2, 1>(A_coltail, B_tailtail, N, D_tail, M_tail, out_coltail, pos_D_round, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 22:  // trailing inputs, outputs: 2, 2
            sgemm_colmajor_narrow_padded<2, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<4, 2>(A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<2, 2>(A_coltail, B_tailtail, N, D_tail, M_tail, out_coltail, pos_D_round, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 31:  // trailing inputs, outputs: 3, 1
            sgemm_colmajor_narrow_padded<3, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<4, 1>(A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<3, 1>(A_coltail, B_tailtail, N, D_tail, M_tail, out_coltail, pos_D_round, A_col_stride, B_col_stride, out_col_stride);
            return;
        case 32:  // trailing inputs, outputs: 3, 2
            sgemm_colmajor_narrow_padded<3, 3>(A_coltail, B_rowtail, N, D_tail, M_round, out, pos_round_mat, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<4, 2>(A, B_coltail, N, D_round, M_tail, out_coltail, false, A_col_stride, B_col_stride, out_col_stride);
            sgemm_colmajor_narrow_padded<3, 2>(A_coltail, B_tailtail, N, D_tail, M_tail, out_coltail, pos_D_round, A_col_stride, B_col_stride, out_col_stride);
            return;
        default:
            assert(false); // switch should be collectively exhaustive
            return;
    }
}
