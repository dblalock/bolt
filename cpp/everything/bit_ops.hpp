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
// #include <type_traits> // just for static_assert
#include <iostream>
#include <string.h> // for ffs()

namespace { // anonymous namespace

// ------------------------ debugging

template<class T> // dumps the bits in logical order (ie, msb always first)
inline void dumpBits(T x, bool newline=true) {
	// for (int i = 0; i < sizeof(x) ; i++) {
	for (int i = sizeof(x) - 1; i >= 0 ; i--) {
		std::cout << " ";
		for (int j = 7; j >= 0; j--) {
			uint64_t mask = ((uint64_t)1) << (8*i + j);
			uint64_t masked = mask & x;
			std::cout << (bool)masked;
		}
	}
	if (newline) { std::cout << "\n"; }
}

template<class T> // dumps the raw bits in memory order
inline void dumpEndianBits(T x, bool newline=true) {
	const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&x);
	for (int i = 0; i < sizeof(x); i++) {
		std::cout << " ";
		const uint8_t byte = *(ptr + i);
		for (int j = 0; j < 8; j++) {
			uint64_t mask = ((uint8_t)1) << j;
			uint64_t masked = mask & byte;
			std::cout << (bool)masked;
		}
	}
	if (newline) { std::cout << "\n"; }
}

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
