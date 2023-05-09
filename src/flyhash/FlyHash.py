from typing import Optional, Union

import numpy as np

__author__ = "TeddyHuang-00"
__copyright__ = "TeddyHuang-00"
__license__ = "MIT"


class FlyHash:
    """FlyHash is a local sensitive hashing (LSH) algorithm,
    based on the paper "A neural algorithm for fundamental computing problems"
    by S. Dasgupta, C. F. Stevens, and S. Navlakha (2017).

    FlyHash is a LSH algorithm that maps input data to a sparse hash embedding,
    where the dimension of the hash embedding is much larger than the input,
    and keeps the locality of the input data in the hash embedding.

    FlyHash is designed to be cheap to compute, yet not ganranteeing
    memory efficiency. It is suitable for hashing small to medium sized data
    (d ~ 10-1000) to a large hash embedding (m ~ 100-10000).

    Example
    -------
    Using a large hash_dim m=100 for a small input_dim d=10:

    >>> import numpy as np
    >>> from flyhash import FlyHash
    >>> d = 10
    >>> m = 100
    >>> flyhash = FlyHash(d, m)
    >>> data = np.random.randn(5, d)
    >>> hashed_data = flyhash(data)

    """

    def __init__(
        self,
        input_dim: int,
        hash_dim: int,
        density: Union[int, float] = 0.1,
        sparsity: float = 0.05,
        quant_step: Optional[float] = None,
        dtype: type = np.int32,
        seed: Optional[int] = None,
    ):
        """Initialize FlyHash object.

        Parameters
        ----------
        input_dim : int
            Input dimension of the data to be hashed.

            Note that FlyHash can only hash data with a fixed input dimensions.
        hash_dim : int
            Output dimension of the hash embeddings.
        density : Union[int, float], optional
            The connection density from input to output,
            either a float between 0 and 1, or an integer greater than 0,
            by default 0.1.

            If `density` is a float, each column of the projection matrix
            has non-zero entries with probability `density`.

            If `density` is an integer, each column of the projection matrix
            has exactly `density` non-zero entries.
        sparsity : float, optional
            The sparsity level of hash embeddings, must be a float
            between 0 and 1, by default 0.05.

            The sparsity level is defined as the fraction of non-zero
            entries in the hash embeddings. The precise number of non-zero
            entries is determined by `hash_dim` and `sparsity` as
            `round(hash_dim * sparsity)`.
        quant_step : Optional[float], optional
            The quantization step size to use, either a float representing
            the step size, or `None` for all-or-none clipping, by default `None`.

            If `quant_step` is a float, the hash embeddings will be quantized
            to an integer multiple of `quant_step`. To avoid information loss,
            the ceiling of the quantized value is used, i.e. the quantized value
            is always greater than 0 iff the original value is greater than 0.
        dtype : type, optional
            Data type of the output hash embeddings, by default np.int32.

            Note that the data type must be one of the integer types.
        seed : Optional[int], optional
            The random seed to use when generating the projection matrix,
            by default `None`.
        """
        self.input_dim = input_dim
        assert self.input_dim > 0, "input_dim must be positive"

        self.hash_dim = hash_dim
        assert self.hash_dim > 0, "hash_dim must be positive"

        self.density = density
        if isinstance(self.density, int):
            assert self.density > 0, "density must be positive"
        else:
            assert 0 < self.density < 1, "density must be between 0 and 1"

        self.sparsity = sparsity
        assert 0 < self.sparsity < 1, "sparsity must be between 0 and 1"
        self.num_winners = int(np.round(self.hash_dim * self.sparsity))

        self.quant_step = quant_step
        if self.quant_step is not None:
            assert self.quant_step > 0, "quant_step must be positive"

        self.dtype = dtype
        assert np.issubdtype(
            self.dtype, np.integer
        ), "dtype must be one of np.integer type"
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

        self.projection_matrix = np.zeros((self.hash_dim, self.input_dim), dtype=bool)
        self._construct_projection_matrix()

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Hash input_data to hash_dim vector(s).

        Parameters
        ----------
        input_data : np.ndarray
            Input data to be hashed, must be 1D (input_dim,)
            or 2D (batch_size, input_dim).

        Returns
        -------
        np.ndarray
            Hashed data, 1D (hash_dim,) or 2D (batch_size, hash_dim)
            according to the shape of input_data.
        """
        assert 0 < input_data.ndim <= 2, "input_data must be 1D or 2D"
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]
        assert input_data.shape[1] == self.input_dim, (
            f"input_data of shape {input_data.shape[0]}x{input_data.shape[1]} has "
            f"input dimension incompatible with {self.input_dim}"
        )
        projected_data = input_data @ self.projection_matrix.T
        return np.apply_along_axis(self._winner_take_all, 1, projected_data).squeeze()

    def _construct_projection_matrix(self):
        """Construct projection matrix from input_dim to hash_dim.

        Notes
        -----
            The projection matrix `W` is a binary matrix of shape (hash_dim, input_dim).

            This method constructs `W` by sampling from a binomial distribution
            with parameters `n=input_dim` and `p=density` if `density` is a float,
            or by sampling from a uniform distribution over `input_dim`
            if `density` is an integer.

            This method is called by the constructor. It SHOULD NOT be called again
            later to avoid overwriting the projection matrix. Doing so will make the
            hash embeddings non-deterministic.
        """
        if isinstance(self.density, int):
            assert self.density <= self.hash_dim, "density must be less than hash_dim"
            # Create fixed number of connections when density is an integer.
            # Each column of the projection matrix has exactly density non-zero entries.
            # The non-zero entries are all 1.
            for idx in range(self.hash_dim):
                self.projection_matrix[
                    idx, self.rng.choice(self.input_dim, self.density, replace=False)
                ] = True
        else:
            # Create variable number of connections when density is a float.
            # Each column of the projection matrix has non-zero entries
            # with probability density.
            # The non-zero entries are all 1.
            self.projection_matrix = self.rng.choice(
                [False, True],
                size=(self.hash_dim, self.input_dim),
                p=[1 - self.density, self.density],
            )

    def _winner_take_all(self, input_vector: np.ndarray) -> np.ndarray:
        """Winner-take-all operation.

        This also includes quantization step using `quant_step` if specified,
        to reduce the number of unique values.

        Parameters
        ----------
        input_vector : np.ndarray
            1D input vector to be winner-take-all-ed. The number of winners
            is pre-determined by `num_winners`.

        Returns
        -------
        np.ndarray
            1D winner-take-all-ed vector of given integer type `dtype`.
        """
        result = np.zeros_like(input_vector, dtype=self.dtype)
        indices = np.argpartition(input_vector, -self.num_winners)[-self.num_winners :]
        if self.quant_step is None:
            # Clip to 0 or 1.
            result[indices] = 1
        else:
            # Quantize to steps as specified by quant_step.
            # Clipped by the range of dtype.
            result[indices] = np.ceil(input_vector[indices] / self.quant_step).clip(
                0, np.iinfo(self.dtype).max
            )
        return result
