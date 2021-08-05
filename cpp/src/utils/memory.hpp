//
//  memory.hpp
//  Dig
//
//  Created by DB on 2016-10-16
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __DIG_MEMORY_HPP
#define __DIG_MEMORY_HPP

#ifdef BLAZE
    #include "src/external/eigen/Eigen/Core"
#else
    #include "Core"  // minimal subset of EIGEN that will compile
#endif

static constexpr int kDefaultAlignBytes = EIGEN_DEFAULT_ALIGN_BYTES;
static_assert(kDefaultAlignBytes == 32, "EIGEN_DEFAULT_ALIGN_BYTES is not 32!");

// ------------------------------------------------ Alignment

template<class T, int AlignBytes, class IntT>
inline constexpr IntT aligned_length(IntT ncols) {
    static_assert(AlignBytes >= 0, "AlignBytes must be nonnegative");
    assert(ncols > 0);
    if (AlignBytes == 0) { return ncols; }

    int16_t align_elements = AlignBytes / sizeof(T);
    int16_t remaindr = ncols % align_elements;
    if (remaindr > 0) {
        ncols += align_elements - remaindr;
    }
    return ncols;
}

// helper struct to get Eigen::Map<> MapOptions based on alignment in bytes
template<int AlignBytes> struct _AlignHelper {};
template<> struct _AlignHelper<0> {
    static constexpr int AlignmentType = Eigen::Unaligned;
};
template<> struct _AlignHelper<kDefaultAlignBytes> {
    static constexpr int AlignmentType = Eigen::Aligned;
};

template<class T, int AlignBytes=kDefaultAlignBytes>
static inline T* aligned_alloc(size_t n) {
    static_assert(AlignBytes == 0 || AlignBytes == kDefaultAlignBytes,
        "Only AlignBytes values of 0 and kDefaultAlignBytes are supported!");
    static_assert(EIGEN_DEFAULT_ALIGN_BYTES == AlignBytes,
        "AlignBytes does not match Eigen default align bytes!");
    if (AlignBytes == kDefaultAlignBytes) {
        return Eigen::aligned_allocator<T>{}.allocate(n);
    } else {
        return new T[n];
    }
}
template<class T, int AlignBytes=kDefaultAlignBytes>
static inline void aligned_free(T* p) {
    static_assert(AlignBytes == 0 || AlignBytes == kDefaultAlignBytes,
        "Only AlignBytes values of 0 and kDefaultAlignBytes are supported!");
    if (AlignBytes == kDefaultAlignBytes) {
        Eigen::aligned_allocator<T>{}.deallocate(p, 0); // 0 unused
    } else {
        delete[] p;
    }
}


#endif // __DIG_MEMORY_HPP
