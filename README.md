
<p align="center">
  <img src="https://github.com/dblalock/bolt/blob/master/assets/bolt.jpg?raw=true" alt="Bolt" width="611px" height="221px"/>
  <!-- <img src="https://github.com/dblalock/bolt/blob/master/assets/bolt.jpg?raw=true" alt="Bolt" width="685px" height="248px"/> -->
</p>

Bolt is an algorithm for compressing vectors of real-valued data and running mathematical operations directly on the compressed representations.

If you have a large collection of vectors and can tolerate lossy compression, Bolt can probably save you 10-200x space. If what you want to do with these vectors is compute dot products and/or Euclidean distances to other vectors, Bolt can probably save you 10-200x compute time as well.

Bolt also has [theoretical guarantees](https://github.com/dblalock/bolt/blob/master/assets/bolt-theory.pdf?raw=true) bounding the errors in its approximations.

NOTE: All the code, documentation, and results associated with Bolt's KDD paper can be found in the `experiments/` directory. See the README therein for details.

NOTE 2: This page is currently under construction, as is the Python API described below.


## Example: Matrix-Vector multiplies

```python
import bolt
import numpy as np
from scipy.stats.stats import pearsonr
import timeit

# dataset -- when we receive a "query" vector q, we want the product Xq
X = np.random.randn(1e6, 256)

# obvious way
def query_received(q):
    return np.dot(X, q)

# faster way with Bolt
encoder = bolt.Encoder().fit(X)  # train a Bolt encoder on this distribution
X_compressed = encoder.transform(X)  # compress the dataset

def query_received_bolt(q):
    return encoder.dot(X_compressed, Q.T)

# massive space savings
print(X.nbytes)  # 1e6 * 256 * 8B = 2.048 GB
print(X_compressed.nbytes)  # 1e6 * 32B = 32 MB

# massive time savings
Q = np.random.randn(1e3, 256)  # queries; need dot products with rows in X
print timeit.Timer('for q in Q: query_received(q)').timeit(3)
print timeit.Timer('for q in Q: query_received_bolt(q)').timeit(3)

# nearly identical dot products (modulo offset/scaling)
print("correlation between true and Bolt dot products: " {}".format(
    pearsonr(out, out_bolt))  # >.9, despite random data being worst case

# recover offset/scaling if needed
out3 = encoder.rescale_output(out_bolt)
```

## How does it work?

Bolt is based on [vector quantization](https://en.wikipedia.org/wiki/Vector_quantization). For details, see the Bolt paper (to be uploaded soon...).

## Benchmarks

Bolt includes a thorough set of speed and accuracy benchmarks. See the `experiments/` directory.


## Example: Kernel Density Estimation


## Example: K-Nearest Neighbor Search


<!--
# obvious way to get dot products
out = np.dot(X, Q.T)

# obvious way to get dot products if we get queries one at a time
for i, q in enumerate(Q):  # would be callbacks, not loop
    out[:, i] = np.dot(X, q)

# faster way: use Bolt approximate dot products
out2 = encoder.dot(X, Q.T)

# even faster way: give Bolt an X that's already compressed
X_compressed = encoder.transform(X)
out3 = encoder .dot(X_compressed, Q.T)
 -->
