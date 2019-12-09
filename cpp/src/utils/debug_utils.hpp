//
//  debug_utils.h
//  Dig
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __debug_utils_hpp
#define __debug_utils_hpp

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include <type_traits>  // for std::is_signed

#ifdef __cplusplus
	#include <iostream>
	#include <cstdarg>
	#include <string>
	#include <cstdint>

	#define PRINT_DEBUG(STR) \
		std::cout << __func__ << "(): " STR << "\n";

	#define PRINT_VAR_DEBUG(VAR) \
		std::cout << __func__ << "(): " #VAR << ": " << VAR << "\n";

	#define PRINT(STR) \
		std::cout << STR << "\n";

	#define PRINT_VAR(VAR) \
		std::cout << #VAR ": " << VAR << "\n";

	#define PRINTLN_VAR(VAR) \
		std::cout << #VAR ":\n" << VAR << "\n";

	template<int MAX_LEN=1024>
	static inline std::string string_with_format(const char* fmt, ...) {
		va_list args;
		va_start(args, fmt);
		char buff[MAX_LEN];
		vsnprintf(buff, MAX_LEN-1, fmt, args);
		return std::string(buff);
		va_end(args);
	}

	template<typename P>
	inline int32_t pretty_ptr(P ptr) {
		return (((int64_t)ptr) << 40) >> 40;
	}


#endif

#ifdef __cplusplus
extern "C" {
#endif

// #ifdef DEBUG
//	#define clean_errno() (errno == 0 ? "None" : strerror(errno))
//	#define log_error(M, ...) fprintf(stderr, "[ERROR] (%s:%d: errno: %s) " M "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)
//	#define assertf(A, M, ...) if(!(A)) {log_error(M, ##__VA_ARGS__); assert(A); }
// #else
// 	#define assertf(A, M, ...)
// #endif

static inline void print_var(const char* name, double val) {
	printf("%s=%g\n", name, val);
}

static inline void print(const char* str) {
	printf("%s\n", str);
}

#define DEBUGF(format, ...) \
    printf("%s(): " format "\n", __func__, ##__VA_ARGS__);

// won't work cuz we don't know size of elements for void*
//inline void print_array(const char* name, void* ar, int len) {
//	double*v = (double*) ar;
//	printf("%s: ", name);
//	for(unsigned i = 0; i < len; ++i) {
//		printf("%g ", v[i]);
//	}
//	printf("\n");
//}

#ifdef __cplusplus
}
#endif

template<class T> // dumps the bits in logical order (ie, msb always first)
inline void dumpBigEndianBits(T x, bool newline=true) {
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

// dumps the raw bits in memory order (little endian within bytes)
// template<class T, class _=typename std::enable_if< !std::is_pointer<T>::value >::type>
inline void dump_bits(const void* x, size_t size, bool newline=true) {
	const uint8_t* ptr = reinterpret_cast<const uint8_t*>(x);
	for (int i = 0; i < size; i++) {
		std::cout << " ";
		for (int j = 0; j < 8; j++) {
			uint64_t mask = ((uint8_t)1) << j;
			uint64_t masked = mask & ptr[i];
			std::cout << (bool)masked;
		}
	}
	if (newline) { std::cout << "\n"; }
}

// dumps the raw bits in memory order (little endian within bytes)
template<class T, class _=typename std::enable_if< !std::is_pointer<T>::value >::type>
inline void dump_bits(T x, bool newline=true) {
	// printf("wtf, why is this getting called...");
	const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&x);
	dump_bits(ptr, sizeof(x), newline);
}

template<class T, class CastToT=T> // dumps the raw bytes in memory order
inline void dump_elements(const T* x, size_t len=1, size_t newline_every=1,
	int rowmark_every=8)
{
	const CastToT* ptr = reinterpret_cast<const CastToT*>(x);
	if (len == 0) { return; }
	size_t len_elements = (len * sizeof(T) / sizeof(CastToT));
	// size_t len_elements = len;
	if (newline_every == 1) {
		newline_every = len_elements >= 32 ? 32 : len_elements;
	} else if (newline_every == 0) {
		newline_every = len + 1;
	}
	// printf("dump_elements: len=%lu, len_elements = %lu\n", len, len_elements);
	// printf("dump_elements: newline_every=%lu\n", newline_every);
	for (size_t i = 0; i < len_elements; i++) {
		if (std::is_signed<CastToT>::value) {
			printf("%4d", (int)ptr[i]);  // wider to allow for negative sign
		} else {
			printf("%3d", (int)ptr[i]);
		}
		// printf("%d", (int)ptr[i]);
		int write_newline = newline_every > 0 && ((i+1) % newline_every) == 0;
		// } else if (((i+1) % 8 == 0) && i + 1 < len_elements) { // write 8B separator unless at very end
		if (write_newline) {
			printf(",\n");
		} else {
			size_t idx_in_row = i % newline_every;
			if ((idx_in_row + 1) % 8) {
				printf(",");
			} else if (i + 1 < len_elements) { // write separator unless at end
				printf(" | ");
			}
		}
		if (rowmark_every > 0 && write_newline && i != (len_elements - 1)) {
			size_t row_idx = (i + 1) / newline_every;
			if (row_idx % rowmark_every == 0) {
				// printf("  row  %d\n", (int)row_idx);
				printf("   #%d\n", (int)row_idx);
			}
		}
	}
	if (newline_every <= len && ((len_elements % newline_every) != 0)) { printf("\n"); }
}
template<class T, class CastToT=T,
	class _=typename std::enable_if< !std::is_pointer<T>::value >::type >
inline void dump_elements(T x, size_t newline_every=1, int rowmark_every=8) {
	dump_elements<T, CastToT>(&x, 1, newline_every, rowmark_every);
}

template<class T> // dumps the raw bytes in memory order
inline void dump_bytes(const T* x, size_t len=1, size_t newline_every=1,
	int rowmark_every=8)
{
	dump_elements<T, uint8_t>(x, len, newline_every, rowmark_every);
}

template<class T, class _=typename std::enable_if< !std::is_pointer<T>::value >::type >
inline void dump_bytes(T x, size_t newline_every=1) {
	dump_bytes((uint8_t*)&x, sizeof(T), newline_every, -1);
}

#ifdef __AVX__
#include <immintrin.h>

template<class CastToT=uint8_t>
inline void dump_m256i(const __m256i& v, bool newline=true) {
	// for (int i = 0; i < 4; i++) {
	// 	// uint64_t x = _mm256_extract_epi64(v, i);
	// 	// dump_elements<
	// 	dump_elements<uint64_t, CastToT>(_mm256_extract_epi64(v, i), -1);
 //        std::cout << " | ";
	// }
	dump_elements<uint64_t, CastToT>(_mm256_extract_epi64(v, 0), -1);
    std::cout << " | ";
	dump_elements<uint64_t, CastToT>(_mm256_extract_epi64(v, 1), -1);
    std::cout << " | ";
	dump_elements<uint64_t, CastToT>(_mm256_extract_epi64(v, 2), -1);
    std::cout << " | ";
	dump_elements<uint64_t, CastToT>(_mm256_extract_epi64(v, 3), -1);
    if (newline) { std::cout << "\n"; }
}

inline void dump_m256i_bits(const __m256i& v, bool newline=true) {
	// for (int i = 0; i < 4; i++) {
	// 	dump_bits(_mm256_extract_epi64(v, i), false);
 //        std::cout << "  ";
	// }
	dump_bits(_mm256_extract_epi64(v, 0), false);
	std::cout << "  ";
	dump_bits(_mm256_extract_epi64(v, 1), false);
	std::cout << "  ";
	dump_bits(_mm256_extract_epi64(v, 2), false);
	std::cout << "  ";
	dump_bits(_mm256_extract_epi64(v, 3), false);
    if (newline) { std::cout << "\n"; }
}

template<class CastToT=uint8_t>
inline void dump_m128i(const __m128i& v, bool newline=true) {
	// for (int i = 0; i < 2; i++) {
	// 	dump_elements<uint64_t, CastToT>(_mm_extract_epi64(v, i), false);
 //        std::cout << "  ";
	// }
	dump_elements<uint64_t, CastToT>(_mm_extract_epi64(v, 0), false);
	std::cout << "  ";
	dump_elements<uint64_t, CastToT>(_mm_extract_epi64(v, 1), false);
    if (newline) { std::cout << "\n"; }
}

inline void dump_m128i_bits(const __m128i& v, bool newline=true) {
	// for (int i = 0; i < 2; i++) {
	// 	dump_bits(_mm_extract_epi64(v, i), false);
 //        std::cout << "  ";
	// }
	dump_bits(_mm_extract_epi64(v, 0), false);
	std::cout << "  ";
	dump_bits(_mm_extract_epi64(v, 1), false);
    if (newline) { std::cout << "\n"; }
}

inline void dump_16B(void* ptr, bool newline=true) {
	dump_m128i(_mm_loadu_si128((__m128i*)ptr), newline);
}

#endif // ifdef AVX

#endif
