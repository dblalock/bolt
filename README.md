
<p align="center">
  <img src="https://github.com/dblalock/bolt/blob/master/assets/bolt.jpg?raw=true" alt="Bolt" width="611px" height="221px"/>
  <!-- <img src="https://github.com/dblalock/bolt/blob/master/assets/bolt.jpg?raw=true" alt="Bolt" width="685px" height="248px"/> -->
</p>

Bolt is an algorithm for compressing vectors of real-valued data and running mathematical operations directly on the compressed representations.

If you have a large collection of mostly-dense vectors and can tolerate lossy compression, Bolt can probably save you 10-200x space and compute time.

Bolt also has [theoretical guarantees](https://github.com/dblalock/bolt/blob/master/assets/bolt-theory.pdf?raw=true) bounding the errors in its approximations.

<!-- NOTE: All the code, documentation, and results associated with Bolt's KDD paper can be found in the `experiments/` directory. See the README therein for details. A cleaned-up version of the paper is available [here](https://github.com/dblalock/bolt/blob/master/assets/bolt.pdf?raw=true). -->

## Installing

Provided that you're on a machine with [AVX2 instructions](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2) (which basically means an Intel/AMD processor from fall 2013 or later), you can just:

```bash
  $ brew install swig  # for wrapping C++; use apt-get, yum, etc, if not OS X
  $ pip install numpy  # pybolt installation needs numpy already present
  $ git clone https://github.com/dblalock/bolt.git
  $ cd bolt && python setup.py install
```

If your machine doesn't have AVX2 instructions, Bolt [doesn't support it yet](https://github.com/dblalock/bolt/issues/2), unfortunately. Contributions welcome.


## How does it work?

Bolt is based on [vector quantization](https://en.wikipedia.org/wiki/Vector_quantization). For details, see the [Bolt paper](https://github.com/dblalock/bolt/blob/master/assets/bolt.pdf?raw=true) or [slides](https://github.com/dblalock/bolt/blob/master/assets/bolt-slides.pdf?raw=true).

## Benchmarks

Bolt includes a thorough set of speed and accuracy benchmarks. See the `experiments/` directory. This is also what you want if you want to reproduce the results in the paper.

Note that all of the timing results use the raw C++ implementation. At present, the Python wrapper is slower. If you're interested in having a full-speed wrapper, let me know and I'll allocate time to making this happen.

## Basic usage
```python
X, queries = some N x D array, some iterable of length D arrays

# these are approximately equal (though the latter are shifted and scaled)
enc = bolt.Encoder(reduction='dot').fit(X)
[np.dot(X, q) for q in queries]
[enc.transform(q) for q in queries]

# same for these
enc = bolt.Encoder(reduction='l2').fit(X)
[np.sum((X - q) * (X - q), axis=1) for q in queries]
[enc.transform(q) for q in queries]

# but enc.transform() is 10x faster or more
```

## Example: Matrix-vector multiplies

```python
import bolt
import numpy as np
from scipy.stats import pearsonr as corr
from sklearn.datasets import load_digits
import timeit

# for simplicity, use the sklearn digits dataset; we'll split
# it into a matrix X and a set of queries Q
X, _ = load_digits(return_X_y=True)
nqueries = 20
X, Q = X[:-nqueries], X[-nqueries:]

enc = bolt.Encoder(reduction='dot', accuracy='lowest') # can tweak acc vs speed
enc.fit(X)

dot_corrs = np.empty(nqueries)
for i, q in enumerate(Q):
    dots_true = np.dot(X, q)
    dots_bolt = enc.transform(q)
    dot_corrs[i] = corr(dots_true, dots_bolt)[0]

# dot products closely preserved despite compression
print "dot product correlation: {} +/- {}".format(
    np.mean(dot_corrs), np.std(dot_corrs))  # > .97

# massive space savings
print(X.nbytes)  # 1777 rows * 64 cols * 8B = 909KB
print(enc.nbytes)  # 1777 * 2B = 3.55KB

# massive time savings (~10x here, but often >100x on larger
# datasets with less Python overhead; see the paper)
t_np = timeit.Timer(
    lambda: [np.dot(X, q) for q in Q]).timeit(5)        # ~9ms
t_bolt = timeit.Timer(
    lambda: [enc.transform(q) for q in Q]).timeit(5)    # ~800us
print "Numpy / BLAS time, Bolt time: {:.3f}ms, {:.3f}ms".format(
    t_np * 1000, t_bolt * 1000)

# can get output without offset/scaling if needed
dots_bolt = [enc.transform(q, unquantize=True) for q in Q]
```

## Example: K-Nearest Neighbor / Maximum Inner Product Search
```python
# search using squared Euclidean distances
# (still using the Digits dataset from above)
enc = bolt.Encoder('l2', accuracy='high').fit(X)
bolt_knn = [enc.knn(q, k_bolt) for q in Q]  # knn for each query

# search using dot product (maximum inner product search)
enc = bolt.Encoder('dot', accuracy='medium').fit(X)
bolt_knn = [enc.knn(q, k_bolt) for q in Q]  # knn for each query
```

<!-- ## Example: Compression and Decompression
```python

``` -->

## Trivia

Bolt stands for "Based On Lookup Tables". <!--  Feel free to use this exciting fact at parties. -->
