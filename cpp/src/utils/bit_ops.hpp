//  bit_ops.hpp
//
//  Dig
//
//  Created by DB on 4/29/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#ifndef __BIT_OPS_HPP
#define __BIT_OPS_HPP

#include <assert.h>
#include <stdint.h>
#include <iostream>
#include <string.h> // for ffs()

namespace { // anonymous namespace

// ------------------------ popcount

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

} // anonymous namespace
#endif // include guard
