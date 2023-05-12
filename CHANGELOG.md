## 1.0.0 (2023-05-12)

### BREAKING CHANGE

- The projection matrix attribute of FlyHash now is an instance of scipy.sparse.csr_matrix, this might break direct operations on it

### Perf

- :zap: Use scipy.sparse for sparse matrix multiplication

## v0.3.0 (2023-05-10)

### Feat

- :sparkles: Add reset method for FlyHash

### Fix

- :white_check_mark: Set random seed for non-deterministic tests

## v0.2.0 (2023-05-09)

## v0.1.0 (2023-05-09)

### Feat

- :sparkles: Add FlyHash class

### Fix

- :bug: Fix compatibility with py3.7

### Refactor

- :recycle: Rename test function
