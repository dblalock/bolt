#!/usr/bin/env python

import numpy as np

import numba
import zstandard as zstd  # pip install zstandard


# ================================================================ Funcs

def nbits_cost(diffs, signed=True):
    """
    >>> [nbits_cost(i) for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]]
    [0, 2, 3, 3, 4, 4, 4, 5, 5]
    >>> [nbits_cost(i) for i in [-1, -2, -3, -4, -5, -7, -8, -9]]
    [1, 2, 3, 3, 4, 4, 4, 5]
    >>> nbits_cost([])
    array([], dtype=int32)
    >>> nbits_cost([0, 2, 1, 0])
    array([0, 3, 2, 0], dtype=int32)
    >>> nbits_cost([0, 2, 1, 3, 4, 0], signed=False)
    array([0, 2, 1, 2, 3, 0], dtype=int32)
    """
    if diffs is None:
        return None

    diffs = np.asarray(diffs, dtype=np.int32)
    if diffs.size == 0:
        return np.array([], dtype=np.int32)

    if not signed:
        assert np.all(diffs >= 0)
        pos_idxs = diffs > 0
        nbits = np.zeros(diffs.shape, dtype=np.int32)
        nbits[pos_idxs] = np.floor(np.log2(diffs[pos_idxs])) + 1
        nbits[~pos_idxs] = 0
        return nbits

    # shape = diffs.shape
    # diffs = diffs.ravel()
    # zero_idxs = (diffs == 0)
    # # nbits[zero_idxs] = 0
    # nbits = np.zeros(len(diffs), dtype=np.int32)
    # diffs = diffs[~zero_idxs]
    # equiv_diffs = np.abs(diffs) + (diffs >= 0).astype(np.int32)  # +1 if < 0
    # # assert np.all(np.abs(diffs) > 0)
    # # assert np.all(equiv_diffs > 0)
    # nbits[~zero_idxs] = np.ceil(np.log2(equiv_diffs)) + 1
    # nbits = np.asarray(nbits, dtype=np.int32)  # next line can't handle scalar
    # assert np.all(nbits >= 0)

    shape = diffs.shape
    diffs = diffs.ravel()
    equiv_diffs = np.abs(diffs) + (diffs >= 0).astype(np.int32)  # +1 if < 0
    nbits = np.ceil(np.log2(equiv_diffs)) + 1
    nbits = np.asarray(nbits, dtype=np.int32)  # next line can't handle scalar
    nbits[diffs == 0] = 0
    assert np.all(nbits >= 0)

    return nbits.reshape(shape) if nbits.size > 1 else nbits[0]  # unpack if scalar


@numba.njit(fastmath=True)
def zigzag_encode(x):
    """
    >>> [zigzag_encode(i) for i in [0,1,-1,2,-2,3,-3]]
    [0, 1, 2, 3, 4, 5, 6]
    >>> zigzag_encode([0,1,-1,2,-2,3,-3])
    array([0, 1, 2, 3, 4, 5, 6], dtype=int32)
    """
    x = np.asarray(x, dtype=np.int32)
    return (np.abs(x) << 1) - (x > 0).astype(np.int32)


@numba.njit(fastmath=True)
def zigzag_decode(x):
    return np.bitwise_xor(x >> 1, -np.bitwise_and(x, 1))


def quantize(X, nbits=16, minval=None, maxval=None):
    minval = np.min(X) if minval is None else minval
    maxval = np.max(X) if maxval is None else maxval

    unsigned_max = (1 << nbits) - 1
    dtype_min = 1 << (nbits - 1)
    scale = float(unsigned_max) / maxval

    X = np.maximum(0, X - minval)
    X = np.minimum(unsigned_max, X * scale)
    X -= dtype_min  # center at 0

    dtype = {16: np.int16, 12: np.int16, 8: np.int8}[nbits]
    return X.astype(dtype)


# ================================================================

def zstd_compress(buff, comp=None):
    comp = zstd.ZstdCompressor() if comp is None else comp
    if isinstance(buff, str):
        buff = bytes(buff, encoding='utf8')
    return comp.compress(buff)


def zstd_decompress(buff, decomp=None):
    decomp = zstd.ZstdDecompressor() if decomp is None else decomp
    return decomp.decompress(decomp)

# ============================================================== sprintz
# except without the predictive coding part because we do that manually;
# we also omit the run-length encoding because the author says that's a
# huge pain to code and won't change the results much for our fast-changing
# time series; also we don't do the grouping thing since it only
# affects the decoding speed (it could affect the ratio slightly if the
# number of variables were really low and not a multiple of 8, but neither
# is the case for us)

# def bitpack_vec(x, nbits_per_element):
#     n = len(x)
#     total_nbits = n * nbits_per_element
#     bitvec = np.zeros(total_nbits, dtype=np.bool)

#     for i, val in enumerate(x):
#         start_idx = i * nbits_per_element
#         for b in range(nbits_per_element):
#             bit = (val >> b) & 1
#             bitvec[start_idx + b] = bit

#     return np.packbits(bitvec)


# def bitunpack(X, nbits_per_element):
#     was_1d = X.ndim == 1
#     X = np.atleast_2d(X)

#     N, D = X.shape
#     ret = np.unpackbits(X, axis=1)
#     if was_1d:
#         ret = ret.squeeze()
#     return ret


# @numba.njit(fastmath=True)
def bitpack(X, nbits_per_element):
    was_1d = X.ndim == 1
    X = np.atleast_2d(X)
    N, D = X.shape

    # orig_elemsz = X.dtype.itemsize
    orig_elemsz_bits = 8 * X.dtype.itemsize
    assert X.dtype in (np.uint8, np.uint16)

    assert X.dtype in (np.uint8, np.uint16)
    if nbits_per_element == orig_elemsz_bits:
        ret = X
    elif X.dtype == np.uint8:
        # print("N, D, nbits: ", N, D, nbits_per_element)
        # shape = X.shape
        X = X.ravel()
        # unpacked = np.unpackbits(X, count=nbits_per_element, bitorder='little', axis=-1)
        unpacked = np.unpackbits(X, bitorder='little', axis=-1)
        # print("unpacked initial shape: ", unpacked.shape)
        unpacked = unpacked.reshape(N * D, 8)[:, :nbits_per_element]
        # print("unpacked new shape: ", unpacked.shape)
        ret = np.packbits(unpacked.reshape(N, -1), axis=1)
        # ret = ret.reshape(N, -1)
        # print("ret.shape: ", ret.shape)

    else:
        # X_low = (X & 0xff)[:, :, np.newaxis]
        # X_high = ((X & 0xff00) >> 8)[:, :, np.newaxis]
        # X_combined = np.concatenate([X_low, X_high], axis=-1)
        # X = X[:, :, np.newaxis]
        # X = np.concatenate([X, X], axis=-1)
        # X[:, :, 0] = X[:, :, 0] & 0xff
        # X[:, :, 1] = (X[:, :, 1] & 0xff00) >> 8
        # X = X.reshape(N, 2 * D).astype(np.uint8)
        X = np.ascontiguousarray(X).view(np.uint8).reshape(N, 2 * D)

        # print("X shape: ", X.shape)
        unpacked = np.unpackbits(X, axis=1, bitorder='little')
        unpacked = unpacked.reshape(N, orig_elemsz_bits, D)
        # unpacked = unpacked[:, ::-1, :]  # low bits in low idxs
        unpacked = np.ascontiguousarray(unpacked[:, :nbits_per_element])
        ret = np.packbits(unpacked.reshape(N, -1))

    # nbits_per_row = D * nbits_per_element
    # bitmat = np.zeros((N, nbits_per_row), dtype=np.uint8)
    # for j in range(D):
    #     col = X[:, j]
    #     start_idx = j * nbits_per_element
    #     for b in range(nbits_per_element):
    #         bit = (col >> b) & 1
    #         bitmat[:, start_idx + b] = bit
    # ret = np.packbits(bitmat, axis=1)

    if was_1d:
        ret = ret.squeeze()
    return ret


@numba.njit(fastmath=True)
def _sprintz_header_sz(headers, header_elem_nbits):
    _, D = headers.shape

    header_row_sz = int(np.ceil(D * header_elem_nbits / 8))

    rows_total_nbits = headers.sum(axis=1)
    # zero_rows = rows_total_nbits == 0
    # header_sz = np.sum(nzero_rows)  # one byte for run length
    # pair_sums = zero_rows +

    header_sz = 0
    prev_was_zero = False
    for row in rows_total_nbits:
        is_zero = row == 0
        if is_zero:
            if prev_was_zero:
                continue
            else:
                header_sz += 1  # start of run
        else:
            header_sz += header_row_sz
        prev_was_zero = is_zero

    return header_sz


# def sprintz_packed_size(X, nbits=None, just_return_sz=False, postproc='zstd'):
def sprintz_packed_size(X, nbits=None, just_return_sz=True, postproc=None):
    if nbits is None:
        nbits = {1: 8, 2: 16}.get(X.dtype.itemsize, 16)

    unsigned_dtype = {8: np.uint8, 16: np.uint16}[nbits]

    window_len = 8
    pad_nrows = X.shape[0] % window_len
    if pad_nrows != 0:
        pad_rows = np.zeros((pad_nrows, X.shape[1]), dtype=X.dtype)
        X = np.vstack([X, pad_rows])
    N, D = X.shape
    if X.dtype.itemsize > 2:  # basically just catching floats
        # print("sprintz: quantizing X...WTF")
        X = quantize(X, nbits=nbits)
    if np.min(X) < 0:
        # print("sprintz: zigzag_encoding X!")
        X = zigzag_encode(X).astype(unsigned_dtype)
    # else:
    #     print("sprintz: not zigzag_encoding X!")

    header_elem_nbits = {8: 3, 16: 4}[nbits]

    X_nbits = nbits_cost(X, signed=False)
    X_nbits = np.asfarray(X_nbits).reshape(N // window_len, window_len, -1)
    block_nbits = X_nbits.max(axis=1).astype(np.uint8)
    block_nbits[block_nbits == (nbits - 1)] = nbits
    headers = block_nbits

    if just_return_sz:
        payload_sz = int(block_nbits.sum() * window_len / 8)

        header_sz = _sprintz_header_sz(headers, header_elem_nbits)
        # print("header sz: ", header_sz)
        return header_sz + payload_sz

    nwindows = N // window_len
    payloads = []
    for i in range(nwindows):
        start_idx = i * window_len
        end_idx = start_idx + window_len
        X_slice = X[start_idx:end_idx]
        for j in range(D):
            col = X_slice[:, j]
            payloads.append(bitpack(col, headers[i, j]))

    headers = bitpack(headers, header_elem_nbits)
    payloads = np.hstack(payloads)

    if postproc is None:
        return headers.nbytes + payloads.nbytes
    elif postproc == 'zstd':
        return len(zstd_compress(headers)) + len(zstd_compress(payloads))

    #     # nbits_slice = nbits_cost(X_slice, signed=False)
    #     nbits_slice = X_nbits[start_idx:end_idx]
    #     max_nbits = nbits_slice.max(axis=0)
    #     headers[i] = np.minimum(max_nbits, nbits - 1)  # 8->7, 16->15
    #     max_nbits[max_nbits == nbits - 1] = nbits  # 7->8, 15->16

    #     for j in range(D):
    #         col = X_slice[:, j]
    #         payloads.append(bitpack(col, max_nbits[j]))

    # headers = bitpack(headers, header_elem_nbits)
    # payloads = np.hstack(payloads)

    # header_bytes = headers.tobytes()
    # # payload_bytes = headers.tobytes()
    # blosc.compress(buff, typesize=elem_sz,
    #                       cname=compressor, shuffle=shuffle)

    #


if __name__ == '__main__':
    import doctest
    doctest.testmod()
