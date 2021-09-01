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
    #include "src/external/eigen/Eigen/Core"
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
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> ColMatrixXf;


// ================================================================
// Classes
// ================================================================

// ------------------------------------------------ Bolt

// BoltEncoder is the the class that maintains state and wraps the
// core Bolt logic
class BoltEncoder {
public:
    BoltEncoder(int nbytes, float scaleby=1.0);
    ~BoltEncoder() = default;

    // TODO since we aren't using this class from the python directly after
    // all, should do RAII instead of requiring set_* to be called before use
    bool set_centroids(const float* X, int m, int n);
    bool set_centroids(const float* X, long m, long n);
    bool set_data(const float* X, int m, int n);
    bool set_data(const float* X, long m, long n);
    void set_offsets(const float* v, int len);
    void set_scale(float a);

    RowVector<uint16_t> dists_sq(const float* q, int len);
    RowVector<uint16_t> dot_prods(const float* q, int len);
    // vector<uint16_t> dists_sq(const float* q, int len);
    // vector<uint16_t> dot_prods(const float* q, int len);

    vector<int64_t> knn_l2(const float* q, int len, int k);
    vector<int64_t> knn_mips(const float* q, int len, int k);

    // for testing
    bool set_codes(const RowMatrix<uint8_t>& codes);
    bool set_codes(const uint8_t* codes, int m, int n);
    // ColMatrixXf centroids() { return _centroids; }
    RowMatrixXf centroids() { return _centroids; }
    RowMatrix<uint8_t> codes() { return _codes; } // might have end padding

//    template<int Reduction> ColMatrix<uint8_t> lut(const float* q, int len);
    void lut_l2(const float* q, int len);
    void lut_dot(const float* q, int len);
    void lut_l2(const RowVector<float>& q);
    void lut_dot(const RowVector<float>& q);

    ColMatrix<uint8_t> get_lut();
    RowVector<float> get_offsets();
    float get_scale();

private:
    // ColMatrix<float> _centroids;
	RowMatrix<float> _centroids;
	RowMatrix<uint8_t> _codes;
    RowVector<float> _offsets;
    ColMatrix<uint8_t> _lut;
    int64_t _ncodes;
    float _scaleby;
	int _nbytes;
};

#endif
