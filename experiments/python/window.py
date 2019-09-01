
# first 3 functions taken from:
# http://www.johnvinyard.com/blog/?p=268

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

# from .arrays import normalizeMat


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(
            'ws cannot be larger than a in any dimension.'
            'a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    return strided.reshape(dim)


def sliding_windows_of_elements(a, ss, ws=None, flatten=False):
    return [sliding_window(row, ss, ws, flatten) for row in a]


def sliding_windows_of_rows(a, ss, ws=None, flatten=True):
    windowsForRows = sliding_windows_of_elements(a, ss, ws, flatten)
    return np.vstack(windowsForRows)


def _compute_from_seq(allSubseqs, n):
    seqLens = np.array(map(lambda subseqs: subseqs.shape[0], allSubseqs))
    startIdxs = np.r_[0, np.cumsum(seqLens)[:-1]]
    endIdxs = np.r_[startIdxs[1:], n]
    fromSeq = np.zeros(n)
    for i in range(len(startIdxs)):
        startIdx, endIdx = startIdxs[i], endIdxs[i]
        fromSeq[startIdx:endIdx] = i
    return fromSeq

# def flattened_subseqs_of_length(seqs, m, norm=None, return_from_seq=False):
#     # TODO should have flags for returning X and allSubseqs, not just fromSeq

#     # each element of seqs is assumed to be a 1D or 2D array
#     origM = m
#     step = 1
#     origDims = len(seqs[0].shape)
#     if origDims > 1:
#         sampleDimensions = np.prod(seqs[0].shape[1:]) # num cols in mat
#         m *= sampleDimensions  # TODO don't enforce stepping in only one direction
#         step *= sampleDimensions
#         for i, seq in enumerate(seqs):
#             seqs[i] = seq.flatten()

#     allSubseqs = sliding_windows_of_elements(seqs, m, step)
#     X = np.asarray(allSubseqs, dtype=np.float).reshape((-1, m)) # -1 = compute it
#     Xnorm = normalizeMat(X, origM, how=norm)

#     if not return_from_seq:
#         return Xnorm, X, allSubseqs

#     fromSeq = _compute_from_seq(allSubseqs, Xnorm.shape[0])
#     return Xnorm, X, allSubseqs, fromSeq


# simple function for common case
def sliding_window_1D(x, windowLen, step=1):
    return sliding_window(x, windowLen, step)


class InputTooSmallException(Exception):
    pass


def extract_conv2d_windows(
        X, filt_shape, strides=(1, 1), flatten_spatial_dims=False,
        flatten_examples_dim=False, padding='valid'):
    # TODO support NCHW format
    orig_X_ndim = X.ndim
    if X.ndim == 3:
        X = X[np.newaxis, ...]
    assert X.ndim == 4
    assert len(filt_shape) == 2
    assert len(strides) in (2, 4)
    filt_shape = int(filt_shape[0]), int(filt_shape[1])
    if filt_shape[0] > X.shape[1]:  # TODO rm after debug
        raise InputTooSmallException(
            "filt_shape[0] ({}) > X.shape[1] ({})".format(
                filt_shape[0], X.shape[1]))
    if filt_shape[1] > X.shape[2]:
        raise InputTooSmallException(
            "filt_shape[1] ({}) > X.shape[2] ({})".format(
                filt_shape[0], X.shape[2]))

    padding = padding.lower()
    assert padding in ('same', 'valid')

    pad_nrows = filt_shape[0] - 1
    pad_ncols = filt_shape[1] - 1
    if padding == 'same' and (pad_nrows > 0 or pad_ncols > 0):
        padded = np.zeros((X.shape[0], X.shape[1] + pad_nrows,
                           X.shape[1] + pad_ncols, X.shape[3]))
        # NOTE: this should mirror the padding used by scipy and tensorflow;
        # however, since their exact behavior is only vaguely documented, it
        # may diverge from their behavior at any time. See the source code for
        # scipy.signal.convolve2d or https://stackoverflow.com/a/38111069
        row_start = int(pad_nrows) // 2
        row_end = row_start + X.shape[1]
        col_start = int(pad_ncols) // 2
        col_end = col_start + X.shape[2]
        # print("padding to shape:", padded.shape)
        # print("padding: data row start, end:", row_start, row_end)
        # print("padding: data col start, end:", col_start, col_end)
        padded[:, row_start:row_end, col_start:col_end, :] = X
        X = padded

    filt_shape = (1, filt_shape[0], filt_shape[1], X.shape[3])
    if len(strides) == 2:
        strides = (1, strides[0], strides[1], X.shape[3])
    windows = sliding_window(X, filt_shape, strides, flatten=False)

    # strip out dims 3 and 4, since these are always 1; dim 3 is filter
    # position across channels (only one position, since doing 2D conv),
    # and dim 4 is all filter data across examples (not actually
    # convolving across examples); e.g., for first 200 examples from
    # MNIST with a 5x5 filter, goes from shape:
    #   (200, 24, 24, 1, 1, 5, 5, 1)
    # to shape:
    #   (200, 24, 24, 5, 5, 1)
    windows = windows.reshape(windows.shape[:3] + windows.shape[5:])

    if flatten_spatial_dims:
        # nexamples x npositions x filt_size
        windows = windows.reshape(X.shape[0], -1, np.prod(filt_shape))
    if flatten_examples_dim:
        windows = windows.reshape(-1, *windows.shape[2:])

    if orig_X_ndim == 3:
        windows = windows.reshape(windows.shape[1:])
    return windows


if __name__ == '__main__':
    A = np.arange(24).reshape((6, 4))
    print(A)
    ws = 3
    ss = 1
    print(sliding_windows_of_rows(A, ws, ss))

