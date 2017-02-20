
![Bolt](/assets/bolt.jpg?raw=true)

Bolt is an algorithm for compressing vectors of real-valued data and running mathematical operations directly on the compressed representations.

If you have a large, real-valued dataset and can tolerate lossy compression, Bolt might be able to save you 100x space and 100x compute time.

NOTE: this project page is currenlty under construction but will soon include all the code, documentation, and results to reproduce Bolt's KDD paper.


## Example

```python
import bolt
import numpy as np

X = np.random.randn(1e6, 256)  # dataset
Q = np.random.randn(1e3, 256)  # queries; need dot products to rows in X

# train a Bolt encoder on this distribution
encoder = bolt.Encoder().fit(X)

# obvious way to get dot products
out = np.dot(X, Q.T)

# faster way: use Bolt approximate dot products
out2 = encoder.dot(X, Q.T)

# even faster way: give Bolt and X that's already compressed
X_compressed = encoder.transform(X)
out3 = encoder.dot(X_compressed, Q.T)

# massive space savings
print(X.nbytes) # 1e6 * 256 * 8B = 2.048 GB
print(X_compressed.nbytes)  # 1e6 * 32B = 32 MB

# nearly identical dot products (modulo offset/scaling)
from scipy.stats.stats import pearsonr
print("correlation between true and Bolt dot products: " {}".format(
    pearsonr(out, out3))  # >.95

# recover offset/scaling if needed
out3 = encoder.rescale_output(out3)
```

## Benchmarks

Bolt includes a thorough set of performance benchmarks.
