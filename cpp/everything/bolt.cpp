

#include <cstring> // for memcpy

#include "bolt.hpp"
#include "public.hpp"  // defines bolt wrapper class
#include "nn_utils.hpp" // for knn_from_dists

// ------------------------------------------------ BoltEncoder impl

// /** Only accepts values of 8,16,24,32, and returns true iff given one of these */
// bool BoltEncoder::set_nbytes(int nbytes) {
//     if (nbytes == 8 || nbytes == 16 || nbytes == 24 || nbytes == 32) {
//         _nbytes = nbytes;
//         return true;
//     }
//     return false;
// }


BoltEncoder::BoltEncoder(int nbytes): _nbytes(nbytes) {
    bool valid = (nbytes == 2 || nbytes == 8 || nbytes == 16 ||
        nbytes == 24 || nbytes == 32);
    if (!valid) {
        printf("ERROR: Received invalid nbytes %d; "
            "must be one of {2, 8, 16, 24, 32}.", nbytes);
        exit(1);
    }
}

bool BoltEncoder::set_centroids(const float* X, int m, int n) {
    _centroids.resize(m, n);
    auto implied_nbytes = m / (2 * 16);
    assert(_nbytes > 0);
    assert(implied_nbytes == _nbytes);

    int ncodebooks = 2 * _nbytes;
    int ncols = n * ncodebooks;

    // allow unrolling the loops for various nbytes; this will get called from
    // python, so this can't get inlined and they can't get unrolled otherwise
    switch (_nbytes) {
        case 2:
            bolt_encode_centroids<2>(X, ncols, _centroids.data());
            return true;
        case 8:
            bolt_encode_centroids<8>(X, ncols, _centroids.data());
            return true;
        case 16:
            bolt_encode_centroids<16>(X, ncols, _centroids.data());
            return true;
        case 24:
            bolt_encode_centroids<24>(X, ncols, _centroids.data());
            return true;
        case 32:
            bolt_encode_centroids<32>(X, ncols, _centroids.data());
            return true;
        default:
            return false;
    }
    return false;
}
bool BoltEncoder::set_centroids(const float* X, long m, long n) {
    return BoltEncoder::set_centroids(X, (int)m, (int)n);
}

bool BoltEncoder::set_data(const float* X, int m, int n) {
    assert(_nbytes > 0);
    assert(m > 0);
    assert(n > 2 * _nbytes); // equal would probably also work, but play safe
    _ncodes = m;
    int64_t nblocks = ceil(m / 32.0);
    _codes.resize(nblocks * 32, _nbytes);
    _codes.bottomRows(32).setZero(); // ensure mem initialized past end of X

    switch (_nbytes) {
        case 2:
            bolt_encode<2>(X, m, n, _centroids.data(), _codes.data());
            return true;
        case 8:
            bolt_encode<8>(X, m, n, _centroids.data(), _codes.data());
            return true;
        case 16:
            bolt_encode<16>(X, m, n, _centroids.data(), _codes.data());
            return true;
        case 24:
            bolt_encode<24>(X, m, n, _centroids.data(), _codes.data());
            return true;
        case 32:
            bolt_encode<32>(X, m, n, _centroids.data(), _codes.data());
            return true;
        default:
            return false;
    }
    return false;
}
bool BoltEncoder::set_data(const float* X, long m, long n) {
    return BoltEncoder::set_data(X, (int)m, (int)n);
}

bool BoltEncoder::set_codes(const uint8_t* codes, int m, int n) {
    auto implied_nbytes = n / 2;
    assert(implied_nbytes == _nbytes);
    assert(m > 0);
    _ncodes = m;
    int64_t nblocks = ceil(m / 32.0);
    Eigen::Map<const RowMatrix<uint8_t> > buff_wrapper(codes, m, n);

    _codes.resize(nblocks * 32, 2 * _nbytes);
    _codes.bottomRows(32).setZero(); // ensure mem initialized past end of X
    _codes.topRows(m) = buff_wrapper;

    return true;
}
bool BoltEncoder::set_codes(const RowMatrix<uint8_t>& codes) {
    return set_codes(codes.data(), (int)codes.rows(), (int)codes.cols());
}

// mostly for debugging; "real" version would need Reducation template arg
ColMatrix<uint8_t> BoltEncoder::lut(RowVector<float> q) {
    static constexpr int ncentroids = 16;
    int ncodebooks = _nbytes * 2;
    int len = static_cast<int>(q.size());

    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);
    auto lut_ptr = lut_out.data();

    // create lookup table and then scan with it
    switch (_nbytes) {
        case 2:
            bolt_lut<2>(q.data(), len, _centroids.data(), lut_ptr);
            break;
        case 8:
            bolt_lut<8>(q.data(), len, _centroids.data(), lut_ptr);
            break;
        case 16:
            bolt_lut<16>(q.data(), len, _centroids.data(), lut_ptr);
            break;
        case 24:
            bolt_lut<24>(q.data(), len, _centroids.data(), lut_ptr);
            break;
        case 32:
            bolt_lut<32>(q.data(), len, _centroids.data(), lut_ptr);
            break;
        default:
            break;
    }
    return lut_out;
}

template<int Reduction=Reductions::DistL2, class dist_t>
void query(const float* q, int len, int nbytes,
    ColMatrix<float> _centroids, RowMatrix<uint8_t> _codes,
    int64_t _ncodes, dist_t* dists)
{
    static constexpr int ncentroids = 16;
    int ncodebooks = nbytes * 2;
    assert(nbytes > 0);
    assert(_codes.rows() > 0);
    assert(_codes.cols() > 0);
    assert(len == (_centroids.cols() * ncodebooks));

    int64_t N = _codes.rows(); // number of codes stored
    int64_t nblocks = ceil(N / 32.0);

    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);
    auto lut_ptr = lut_out.data();
    assert(lut_ptr != nullptr);

    auto codes_ptr = _codes.data();
    assert(codes_ptr != nullptr);

    // create lookup table and then scan with it
    switch (nbytes) {
        case 2:
            bolt_lut<2, Reduction>(q, len, _centroids.data(), lut_ptr);
            bolt_scan<2>(_codes.data(), lut_out.data(), dists, nblocks);
            break;
        case 8:
            bolt_lut<8, Reduction>(q, len, _centroids.data(), lut_ptr);
            bolt_scan<8>(_codes.data(), lut_out.data(), dists, nblocks);
            break;
        case 16:
            bolt_lut<16, Reduction>(q, len, _centroids.data(), lut_ptr);
            bolt_scan<16>(_codes.data(), lut_out.data(), dists, nblocks);
            break;
        case 24:
            bolt_lut<24, Reduction>(q, len, _centroids.data(), lut_ptr);
            bolt_scan<24>(_codes.data(), lut_out.data(), dists, nblocks);
            break;
        case 32:
            bolt_lut<32, Reduction>(q, len, _centroids.data(), lut_ptr);
            bolt_scan<32>(_codes.data(), lut_out.data(), dists, nblocks);
            break;
        default:
            break;
    }
}

// TODO version that writes to argout array to avoid unnecessary copy
// TODO allow specifying safe (no overflow) scan
template<int Reduction=Reductions::DistL2>
vector<uint16_t> query_all(const float* q, int len, int nbytes,
    ColMatrix<float> _centroids, RowMatrix<uint8_t> _codes, int64_t _ncodes)
{
    RowVector<uint16_t> dists(_codes.rows()); // need 32B alignment, so can't use stl vector
    query(q, len, nbytes, _centroids, _codes, _ncodes, dists.data());

    // copy computed distances into a vector to return; would be nice
    // to write directly into the vector, but can't guarantee 32B alignment
    // without some hacks
    vector<uint16_t> ret(_ncodes);
    // ret.reserve(_ncodes);
    std::memcpy(ret.data(), dists.data(), _ncodes * sizeof(uint16_t));
//    printf("ncodes, ret.size(), %lld, %lld\n", _ncodes, ret.size());
    return ret;
}

template<int Reduction=Reductions::DistL2>
vector<int64_t> query_knn(const float* q, int len, int nbytes,
    ColMatrix<float> _centroids, RowMatrix<uint8_t> _codes,
    int64_t _ncodes, int k)
{
    RowVector<uint16_t> dists(_codes.rows()); // need 32B alignment, so can't use stl vector
    query(q, len, nbytes, _centroids, _codes, _ncodes, dists.data());

    // extract and return the k nearest neighbors
    vector<Neighbor> neighbors = nn::knn_from_dists(dists.data(), _codes.rows(), k);
    vector<int64_t> ret(k);
    for (int i = 0; i < k; i++) {
        ret[k] = static_cast<int64_t>(neighbors[i].idx);
    }
    return ret;
}


vector<uint16_t> BoltEncoder::dists_l2(const float* q, int len) {
    return query_all<Reductions::DistL2>(
        q, len, _nbytes, _centroids, _codes, _ncodes);
}
vector<uint16_t> BoltEncoder::dot_prods(const float* q, int len) {
    return query_all<Reductions::DotProd>(
        q, len, _nbytes, _centroids, _codes, _ncodes);
}

vector<int64_t> BoltEncoder::knn_l2(const float* q, int len, int k) {
    return query_knn<Reductions::DistL2>(
        q, len, _nbytes, _centroids, _codes, _ncodes, k);
}
vector<int64_t> BoltEncoder::knn_mips(const float* q, int len, int k) {
    return query_knn<Reductions::DotProd>(
        q, len, _nbytes, _centroids, _codes, _ncodes, k);
}
