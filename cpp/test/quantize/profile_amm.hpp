//
//  profile_amm.hpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef profile_amm_h
#define profile_amm_h

#ifdef BLAZE
    #include "test/quantize/amm_common.hpp"
#else
    #include "amm_common.hpp"
#endif

namespace {
// ================================================================ mithral

template<class InputT>
struct mithral_amm_task {
    using traits = mithral_input_type_traits<InputT>;
    using scale_t = typename traits::encoding_scales_type;
    using offset_t = typename traits::encoding_offsets_type;
    using output_t = typename traits::output_type;
    static constexpr int scan_block_nrows = 32;
    static constexpr int ncentroids = 16;
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int max_splitvals = 1 << 4;

    mithral_amm_task(int N, int D, int M, int ncodebooks,
                     float lut_work_const):
        N_padded(N % scan_block_nrows == 0 ? N :
            N + (scan_block_nrows - (N % scan_block_nrows))),
        centroids(ncentroids * ncodebooks, D),
        nsplits(ncodebooks * nsplits_per_codebook),
        splitdims(nsplits),
        splitvals(max_splitvals, nsplits),
        encode_scales(nsplits),
        encode_offsets(nsplits),
        nnz_per_centroid(lut_work_const * D / ncodebooks),
        idxs(ncodebooks, nnz_per_centroid),
        amm(N_padded, D, M, ncodebooks, centroids.data(),
            splitdims.data(), splitvals.data(),
            encode_scales.data(), encode_offsets.data(),
            idxs.data(), nnz_per_centroid),
        X(N_padded, D),
        Q(D, M)
    {
        centroids.setRandom();
        splitdims.setRandom();
        for (int i = 0; i < splitdims.size(); i++) {
            splitdims(i) = splitdims(i) % D;
        }
        splitvals.setRandom();
        encode_scales.setRandom();
        encode_offsets.setRandom();

        // randomly initialize idxs, ensuring all are unique and < D
        idxs.setRandom();
        int all_idxs[D];
        for (int i = 0; i < D; i++) {
            all_idxs[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());  // why can't shuffle just create its own...
        for (int c = 0; c < ncodebooks; c++) {  // random sequential idxs
            std::shuffle(all_idxs, all_idxs + D, g);
            std::sort(all_idxs, all_idxs + nnz_per_centroid);
            for (int j = 0; j < nnz_per_centroid; j++) {
                idxs(c, j) = all_idxs[j];
            }
        }

        X.setRandom();
        Q.setRandom();
    }

    void encode() { amm.encode(X.data()); }
    void lut() { amm.lut(Q.data()); }
    void scan() { amm.scan(); }

    void run_matmul(bool create_lut=true) {
        encode();
        if (create_lut) {
            lut();
        }
        scan();
    }

    const ColMatrix<output_t>& output() const { return amm.out_mat; }

    // stuff we pass into the amm object (would be learned during training)
    int N_padded;
    ColMatrix<float> centroids;
    int nsplits;
    RowVector<uint32_t> splitdims;
    ColMatrix<int8_t> splitvals;
    RowVector<scale_t> encode_scales;
    RowVector<offset_t> encode_offsets;
    int nnz_per_centroid;
    RowMatrix<int> idxs;

    // amm object
    mithral_amm<InputT> amm;

    // random data
    ColMatrix<InputT> X;
    ColMatrix<float> Q;
};

template<class InputT=float>
void _profile_mithral(const char* dset_name, uint32_t N, uint32_t D, uint32_t M,
                      int ncodebooks, float lut_work_const=2) {
    mithral_amm_task<InputT> task(N, D, M, ncodebooks, lut_work_const);
    // mithral_amm_task<InputT> task_dense(N, D, M, ncodebooks, -1);

    printf("---- ncodebooks=%d\n", ncodebooks);

    std::string msg;
    auto dtype_str = input_type_traits<InputT>{}.name;

    // auto fmt = "%7s, %3s, %22s, N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t";
    auto fmt_as_cppstring = string_with_format(
        "%s, %-3s, %%-22s, N D M C lut_work_coef:,"
        "%6d, %3d, %3d, %2d, %%4.1f,\t", dset_name, dtype_str,
        N, D, M, ncodebooks);
    auto fmt = fmt_as_cppstring.c_str();
    // printf("fmt string: %s\n", fmt.c_str());
    // fmt = string_with_format()

    // overall AMM time
    msg = string_with_format(fmt, "amm mithral nolut", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.run_matmul(false));

    msg = string_with_format(fmt, "amm mithral sparselut", lut_work_const);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.run_matmul(true));
    // msg = string_with_format( // time if lut already created
    //     "%3s amm mithral nolut      N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
        // "%3s amm mithral nolut      N, D, M, C:\t\t\t\t\t"
        //     "%6d, %3d, %3d, %2d\t\t",
        // dtype_str, N, D, M, ncodebooks);
    // msg = string_with_format(fmt, dset_name, "amm mithral nolut",
    //     dtype_str, N, D, M, ncodebooks, -1.f);

    // using dense centroids, which slows down LUT creation
    auto orig_nnz_per_centroid = task.amm.nnz_per_centroid;
    task.amm.nnz_per_centroid = -1;
    // msg = string_with_format(  // still overall AMM time
    //     "%3s amm mithral denselut   N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    // msg = string_with_format(fmt, dset_name, "amm mithral denselut",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    msg = string_with_format(fmt, "amm mithral denselut", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.run_matmul(true));
    // msg = string_with_format(  // just time to create lut with dense centroids
    //     "%3s amm mithral lut dense  N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    // msg = string_with_format(fmt, dset_name, "amm mithral lut dense",
    //     dtype_str, N, D, M, ncodebooks, -1.f);
    msg = string_with_format(fmt, "amm mithral lut dense", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.lut());
    task.amm.nnz_per_centroid = orig_nnz_per_centroid;

    // back to sparse centroids
    // msg = string_with_format(
    //     "%3s amm mithral lut sparse N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    // msg = string_with_format(fmt, dset_name, "amm mithral lut sparse",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    msg = string_with_format(fmt, "amm mithral lut sparse", lut_work_const);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.lut());
    // msg = string_with_format(
    //     "%3s amm mithral enc        N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    // msg = string_with_format(fmt, dset_name, "amm mithral enc",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    msg = string_with_format(fmt, "amm mithral enc", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.encode());
    // msg = string_with_format(
    //     "%3s amm mithral scan       N, D, M, C, lut_work_coef:\t"
    //         "%6d, %3d, %3d, %2d, %.1f\t",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    // msg = string_with_format(fmt, dset_name, "amm mithral scan",
    //     dtype_str, N, D, M, ncodebooks, lut_work_const);
    msg = string_with_format(fmt, "amm mithral scan", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        task.output().data(), task.output().size(),
        task.scan());
}

template<class InputT=float>
void _profile_mithral(const MatmulTaskShape& shape, std::vector<int> ncodebooks,
                      float lut_work_const=2)
{
    auto dtype_name = input_type_traits<InputT>{}.name;
    printf("------------------------ %s %s\n", shape.name, dtype_name);
    for (auto c : ncodebooks) {
        _profile_mithral<InputT>(
            shape.name, shape.N, shape.D, shape.M, c, lut_work_const);
    }
}

// ================================================================ gemm

template<class MatrixT1, class MatrixT2, class MatrixT3>
void _run_matmul(const MatrixT1& X, const MatrixT2& Q, MatrixT3& out) {
   out.noalias() = X * Q;
}

template<class MatrixT1, class MatrixT2, class MatrixT3>
void _run_our_matmul(const MatrixT1& X, const MatrixT2& Q, MatrixT3& out) {
    // not actually faster than the eigen one
    sgemm_colmajor(
        X.data(), Q.data(), (int)X.rows(), (int)X.cols(), (int)Q.cols(), out.data());
}

void _profile_matmul(const char* dset_name, uint32_t N, uint32_t D, uint32_t M)
{
    using MatrixT = ColMatrix<float>;

    // create random data
    MatrixT X(N, D);
    X.setRandom();
    MatrixT W(D, M);
    W.setRandom();

    // create output matrix to avoid malloc
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M:,   %6d, %3d, %3d,\t\t\t", dset_name, N, D, M);
    auto fmt = fmt_as_cppstring.c_str();

    // time it
    {
        // std::string msg = string_with_format(
        //     "blas matmul               N, D, M:    %6d, %3d, %3d \t\t\t",
        //     orig_N, orig_D, orig_M);
        msg = string_with_format(fmt, "blas matmul");
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            out.data(), out.size(),
            _run_matmul(X, W, out));
    }
    {
        // std::string msg = string_with_format(
        //     "our  matmul               N, D, M:    %6d, %3d, %3d \t\t\t",
        //     orig_N, orig_D, orig_M);
        msg = string_with_format(fmt, "our matmul");
        REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
            out.data(), out.size(),
            _run_our_matmul(X, W, out));
    }
}

template<class MatrixT>
void _run_matmul_fixedW(const MatrixT& X,
                        const MatrixT& W0, MatrixT& sketch_out,
                        const MatrixT& W1, MatrixT& out)
{
   sketch_out.noalias() = X * W0;
   out.noalias() = sketch_out * W1;
}

template<class MatrixT>
void _run_our_matmul_fixedW(const MatrixT& X,
                            const MatrixT& W0, MatrixT& sketch_out,
                            const MatrixT& W1, MatrixT& out)
{
    auto N = (int)X.rows();
    auto D = (int)X.cols();
    auto M = (int)out.cols();
    auto d = (int)W0.cols();
    sgemm_colmajor(X.data(), W0.data(), N, D, d, sketch_out.data());
    sgemm_colmajor(sketch_out.data(), W1.data(), N, d, M, out.data());
}

void _profile_sketch_matmul_fixedW(const char* dset_name, uint32_t N,
    uint32_t D, uint32_t M, uint32_t d)
{
    using MatrixT = ColMatrix<float>;

    // create random matrices of the appropriate sizes
    MatrixT X(N, D); X.setRandom();
    MatrixT W0(D, d); W0.setRandom();
    MatrixT W1(d, M); W1.setRandom();

    // create output matrices to avoid malloc
    MatrixT sketch_out(N, d);
    sketch_out.setRandom();
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M d:, %6d, %3d, %3d, %3d,\t", dset_name, N, D, M, d);
    auto fmt = fmt_as_cppstring.c_str();

    // time it
    // msg = string_with_format("blas sketch fixedW matmul N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "blas sketch fixedW matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_matmul_fixedW(X, W0, sketch_out, W1, out));
    // msg = string_with_format("our  sketch fixedW matmul N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "our sketch fixedW matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_our_matmul_fixedW(X, W0, sketch_out, W1, out));
}

template<class MatrixT>
void _run_sketch_matmul(const MatrixT& X, const MatrixT& W, const MatrixT& S,
                        MatrixT& X_sketched, MatrixT& W_sketched, MatrixT& out)
{
   X_sketched.noalias() = X * S;
   W_sketched.noalias() = S.transpose() * W;
   out.noalias() = X_sketched * W_sketched;
}

template<class MatrixT>
void _run_our_sketch_matmul(
    const MatrixT& X, const MatrixT& W, const MatrixT& S, const MatrixT& St,
    MatrixT& X_sketched, MatrixT& W_sketched, MatrixT& out)
{
    auto N = (int)X.rows();
    auto D = (int)X.cols();
    auto M = (int)W.cols();
    auto d = (int)S.cols();
    sgemm_colmajor(X.data(), S.data(), N, D, d, X_sketched.data());
    sgemm_colmajor(St.data(), W.data(), d, D, M, W_sketched.data());
    sgemm_colmajor(X_sketched.data(), W_sketched.data(), N, d, M, out.data());
}

void _profile_sketch_matmul(const char* dset_name, uint32_t N, uint32_t D,
    uint32_t M, uint32_t d)
{
    using MatrixT = ColMatrix<float>;

    // create random matrices of the appropriate sizes
    MatrixT X(N, D); X.setRandom();
    MatrixT W(D, M); W.setRandom();
    MatrixT S(D, d); S.setRandom();
    MatrixT St(S.transpose());

    // create output matrices to avoid malloc
    MatrixT sketch_X(N, d);
    sketch_X.setRandom();
    MatrixT sketch_W(d, M);
    sketch_W.setRandom();
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M d:, %6d, %3d, %3d, %3d,\t", dset_name, N, D, M, d);
    auto fmt = fmt_as_cppstring.c_str();

    // time it
    // msg = string_with_format("blas sketch matmul        N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "blas sketch matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_sketch_matmul(X, W, S, sketch_X, sketch_W, out));
    // msg = string_with_format("our  sketch matmul        N, D, M, d: %6d, %3d, %3d, %3d \t",
    //     N, D, M, d);
    msg = string_with_format(fmt, "our sketch matmul");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        _run_our_sketch_matmul(X, W, S, St, sketch_X, sketch_W, out));
}

void _profile_matmul_methods(std::vector<int> dvals, MatmulTaskShape shape) {
    auto N = shape.N;
    auto D = shape.D;
    auto M = shape.M;
    printf("------------------------ %s\n", shape.name);
    for (auto d : dvals) {
        _profile_sketch_matmul(shape.name, N, D, M, d);
        _profile_sketch_matmul_fixedW(shape.name, N, D, M, d);
    }
    _profile_matmul(shape.name, N, D, M);
}

// ================================================================ osnap

template<bool SketchW, class SketchT, class ColMatrixT>
void _run_fancy_sketch_matmul(
    const SketchT& sketch, const ColMatrixT& X, const ColMatrixT& Wt,
    ColMatrixT& X_sketched, ColMatrixT& Wt_sketched, ColMatrixT& out)
{
    // printf("\nsketching X\n");
    sketch(X, X_sketched);
    if (SketchW) {
        // printf("sketching W\n");
        // sketch(W, W_sketched, true /*transpose*/);
        sketch(Wt, Wt_sketched);
    }
    // no option to use our gemm here since it would require transposing W
    out.noalias() = X_sketched * Wt_sketched.transpose();
    // if (UseOurGemm) {
    //     auto N = (int)X_sketched.rows();
    //     auto d = (int)X_sketched.cols();
    //     auto M = (int)W_sketched.cols();
    //     sgemm_colmajor(X_sketched.data(), W_sketched.data(),
    //                    N, d, M, out.data());
    // } else {
    //     out.noalias() = X_sketched * W_sketched;
    // }
}

void _profile_osnap(const char* dset_name, uint32_t N, uint32_t D,
                    uint32_t M, uint32_t d, int nsketches)
{
    using MatrixT = ColMatrix<float>;
    MatrixT X(N, D); X.setRandom();
    MatrixT Wt(M, D); Wt.setRandom();

    // create output matrices to avoid malloc
    MatrixT sketch_X(N, d);
    sketch_X.setRandom();
    // MatrixT sketch_Wt(d, M);
    MatrixT sketch_Wt(M, d);
    sketch_Wt.setRandom();
    MatrixT out(N, M);
    out.setRandom();

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, %%-25s, N D M d, s:, %6d, %3d, %3d, %3d, %2d\t",
        dset_name, N, D, M, d, nsketches);
    auto fmt = fmt_as_cppstring.c_str();

    auto sketch = OsnapSketch(D, d, nsketches);

    //
    // sketching W takes almost no time, even for cifar100, so just
    // report fixedW resuls to err on side of optimism and halve the
    // execution time
    //
    // msg = string_with_format(fmt, "osnap");
    // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
    //     out.data(), out.size(),
    //     (_run_fancy_sketch_matmul<true>(
    //         sketch, X, Wt, sketch_X, sketch_Wt, out)));

    msg = string_with_format(fmt, "osnap fixedW");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        out.data(), out.size(),
        (_run_fancy_sketch_matmul<false>(
            sketch, X, Wt, sketch_X, sketch_Wt, out)));
}

void _profile_osnap(std::vector<int> dvals, std::vector<int> nsketches,
                    MatmulTaskShape shape)
{
    // assert(false); // are we in release mode?
    auto N = shape.N;
    auto D = shape.D;
    auto M = shape.M;
    printf("------------------------ %s\n", shape.name);
    for (auto d : dvals) {
        for (auto s : nsketches) {
            if (s > d) { continue; }
            _profile_osnap(shape.name, N, D, M, d, s);
        }
    }
}

// ================================================================ bolt

template<int M, bool Safe=false, class dist_t=void>
void _bolt_query(const uint8_t* codes, int nblocks,
    const float* q, int ncols,
    const float* centroids,
    uint8_t* lut_out, dist_t* dists_out)
{
    // TODO use version of lut that requires offsets and scales
    bolt_lut<M, Reductions::DotProd>(q, ncols, centroids, lut_out);
    bolt_scan<M, Safe>(codes, lut_out, dists_out, nblocks);
}

// template<int ncodebooks, bool encode=false>
template<int ncodebooks, bool encode=true>
void _amm_bolt(const float* X, int nrowsX, const float* Q, int nrows, int ncols,
                     const float* centroids,
                     uint8_t* lut_out, uint16_t* dists_out,
                     uint8_t* codes, int nblocks)
{
    static constexpr int nbytes = ncodebooks / 2;
    // in contrast to multisplit, this precomputes encodings and computes
    // new LUTs when a query comes in, instead of the reverse

    if (encode) {
        bolt_encode<nbytes>(X, nrowsX, ncols, centroids, codes);
    }

    auto q_ptr = Q;
    auto dists_ptr = dists_out;
    for (int i = 0; i < nrows; i++) {  // rows in query matrix, not codes
        _bolt_query<nbytes, true>(
            codes, nblocks, q_ptr, ncols, centroids, lut_out, dists_ptr);
        q_ptr += ncols;
        dists_ptr += nblocks * 32;
    }
}

template<int ncodebooks>
void _template_profile_bolt_amm(const char* dset_name, uint32_t N, uint32_t D,
                                uint32_t M)
{
    static constexpr uint8_t ncentroids = 16;
    auto orig_M = M;
    auto orig_D = D;

    auto nblocks = (M + 31) / 32;
    M = 32 * nblocks;
    if (D % ncodebooks) {  // ensure that ncodebooks evenly divides D
        D += (ncodebooks - (D % ncodebooks));
    }

    // stuff just for encoding; for bolt, we encode the smaller matrix since
    // encoding is slower than lut creation
    RowMatrix<float> X(M, D); X.setRandom();

    // stuff for LUT creation
    ColMatrix<float> centroids(ncentroids, D);          centroids.setRandom();
    RowMatrix<float> Q(N, D);                           Q.setRandom();
    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks); lut_out.setRandom();
    RowVector<float> offsets(D);                        offsets.setRandom();
    float scaleby = 3; // arbitrary number

    // additional stuff for distance computation
    ColMatrix<uint8_t> codes_(M, ncodebooks / 2); codes_.setRandom();
    ColMatrix<uint8_t> codes = codes_.unaryExpr([=](const uint8_t x) {
        return static_cast<uint8_t>(x % ncentroids); });
    ColMatrix<uint16_t> dists_u16(N, M);

    std::string msg;
    auto fmt_as_cppstring = string_with_format(
        "%s, f32, %%-22s, N D M C:,"
        "%6d, %3d, %3d, %2d,\t", dset_name, N, orig_D, orig_M, ncodebooks);
    auto fmt = fmt_as_cppstring.c_str();

    msg = string_with_format(fmt, "amm bolt");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u16.data(), dists_u16.size(),
        (_amm_bolt<ncodebooks>(X.data(), M, Q.data(), N, D, centroids.data(),
            lut_out.data(), dists_u16.data(), codes.data(), nblocks)));

    msg = string_with_format(fmt, "amm bolt noenc");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
        dists_u16.data(), dists_u16.size(),
        (_amm_bolt<ncodebooks, false>(
            X.data(), M, Q.data(), N, D, centroids.data(),
            lut_out.data(), dists_u16.data(), codes.data(), nblocks)));
}

void _profile_bolt_amm(const char* dset_name, uint32_t N, uint32_t D,
                       uint32_t M, int ncodebooks)
{
    if (ncodebooks > D) { return; }
    switch(ncodebooks) {
        case 2: _template_profile_bolt_amm<2>(dset_name, N, D, M); break;
        case 4: _template_profile_bolt_amm<4>(dset_name, N, D, M); break;
        case 8: _template_profile_bolt_amm<8>(dset_name, N, D, M); break;
        case 16: _template_profile_bolt_amm<16>(dset_name, N, D, M); break;
        case 32: _template_profile_bolt_amm<32>(dset_name, N, D, M); break;
        case 64: _template_profile_bolt_amm<64>(dset_name, N, D, M); break;
        default: break;
    }
}

void _profile_bolt_amm(const MatmulTaskShape& shape,
                       std::vector<int> ncodebooks)
{
    printf("------------------------ %s f32\n", shape.name);
    for (auto c : ncodebooks) {
        _profile_bolt_amm(
            shape.name, shape.N, shape.D, shape.M, c);
    }
}



} // anonymous namespace
#endif /* profile_amm_h */
