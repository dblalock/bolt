//
//  neighbors.hpp
//  Dig
//
//  Created by DB on 10/2/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef Dig_Neighbors_hpp
#define Dig_Neighbors_hpp

#include <memory>
#include <vector>

#ifdef BLAZE
    #include "src/external/eigen/Core"
    #include "src/utils/eigen_utils.hpp"
#else
    #include "Core"
    #include "eigen_utils.hpp"
#endif

using std::vector;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::MatrixXi;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
	Eigen::RowMajor> RowMatrixXd;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
	Eigen::RowMajor> RowMatrixXf;
// typedef Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic,
// 	Eigen::RowMajor> MatrixXi;

namespace nn {
	using idx_t = int64_t;
	using dist_t = float;
	namespace {
		static constexpr dist_t kMaxDist = std::numeric_limits<dist_t>::max();
	}
}

// ------------------------------------------------ Neighbor
typedef struct Neighbor {
	typedef nn::idx_t idx_t;
	typedef nn::dist_t dist_t;

	idx_t idx;
	dist_t dist;

	Neighbor() = default;
	Neighbor(const Neighbor& rhs) = default;
	// Neighbor(float d, idx_t i):  idx(i), dist(static_cast<dist_t>(d)) {}
	Neighbor(idx_t i, int d):  idx(i), dist(static_cast<dist_t>(d)) {
		if (dist <= 0) { dist = nn::kMaxDist; }
	}
	Neighbor(idx_t i, float d):  idx(i), dist(static_cast<dist_t>(d)) {
		if (dist <= 0) { dist = nn::kMaxDist; }
	}
	// Neighbor(double d, idx_t i): idx(i), dist(static_cast<dist_t>(d)) {}
	Neighbor(idx_t i, double d): idx(i), dist(static_cast<dist_t>(d)) {
		if (dist <= 0) { dist = nn::kMaxDist; }
	}

} Neighbor;

// ================================================================
// Classes
// ================================================================

// ------------------------------------------------ Bolt

// BoltEncoder is the the class that maintains state and wraps the
// core Bolt logic
class BoltEncoder {
public:
    BoltEncoder(int nbytes);
    ~BoltEncoder() = default;

    // bool set_nbytes(int nbytes);
    bool set_centroids(const float* X, int m, int n);
    bool set_centroids(const float* X, long m, long n);
    bool set_data(const float* X, int m, int n);
    bool set_data(const float* X, long m, long n);

    vector<uint16_t> dists_l2(const float* q, int len);
    vector<uint16_t> dot_prods(const float* q, int len);

    vector<int64_t> knn_l2(const float* q, int len, int k);
    vector<int64_t> knn_mips(const float* q, int len, int k);

    bool set_codes(const RowMatrix<uint8_t>& codes);
    bool set_codes(const uint8_t* codes, int m, int n);

    // for testing
    ColMatrix<float> centroids() { return _centroids; }
    RowMatrix<uint8_t> codes() { return _codes; } // might have end padding
    ColMatrix<uint8_t> lut(RowVector<float> q);

private:
	ColMatrix<float> _centroids;
	RowMatrix<uint8_t> _codes;
	int64_t _ncodes;
	int _nbytes;
};

#endif
