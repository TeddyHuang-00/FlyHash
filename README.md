[![ReadTheDocs](https://readthedocs.org/projects/FlyHash/badge/?version=latest)](https://FlyHash.readthedocs.io/en/latest/)
[![Coveralls](https://img.shields.io/coveralls/github/TeddyHuang-00/FlyHash/main.svg)](https://coveralls.io/r/TeddyHuang-00/FlyHash)
[![PyPI-Server](https://img.shields.io/pypi/v/FlyHash.svg)](https://pypi.org/project/FlyHash/)
[![Monthly Downloads](https://pepy.tech/badge/FlyHash/month)](https://pepy.tech/project/FlyHash)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# FlyHash

> A novel hashing algorithm based on "A neural algorithm for a fundamental computing problem" by S. Dasgupta, C. F. Stevens, and S. Navlakha (2017)

FlyHash is a LSH algorithm that maps input data to a sparse hash embedding,
where the dimension of the hash embedding is much larger than the input,
and keeps the locality of the input data in the hash embedding.

FlyHash is designed to be cheap to compute, yet not ganranteeing
memory efficiency. It is suitable for hashing small to medium sized data
($d$ ~ 10-1000) to a large hash embedding ($m$ ~ 100-10000).

## Usage

Using a large hash_dim $m=100$ for a small input_dim $d=10$:

```python-repl
>>> import numpy as np
>>> from flyhash import FlyHash
>>> d = 10
>>> m = 100
>>> flyhash = FlyHash(d, m)
>>> data = np.random.randn(5, d)
>>> hashed_data = flyhash(data)
```

For detailed usage, please refer to the [documentation](https://FlyHash.readthedocs.io/en/latest/).
