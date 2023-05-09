from typing import Optional, Union

import numpy as np

__author__ = "TeddyHuang-00"
__copyright__ = "TeddyHuang-00"
__license__ = "MIT"


class FlyHash:
    """
    FlyHash is a python package for local sensitive hashing (LSH),
    based on the paper "A neural algorithm for fundamental computing problems"
    by S. Dasgupta, C. F. Stevens, and S. Navlakha (2017).
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
        """
        Initialize FlyHash object.

        :param input_dim: input dimension
        :param hash_dim: hash dimension
        :param density: density of the projection matrix
        :param sparsity: sparsity of the hash code
        :param quant_step: quantization step
        :param dtype: data type of the hash code
        :param seed: random seed
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
        self.__construct_projection_matrix()

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """
        Hash input_data to hash_dim vector(s).
        """
        assert 0 < input_data.ndim <= 2, "input_data must be 1D or 2D"
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]
        assert input_data.shape[1] == self.input_dim, (
            f"input_data of shape {input_data.shape[0]}x{input_data.shape[1]} has "
            f"input dimension incompatible with {self.input_dim}"
        )
        projected_data = input_data @ self.projection_matrix.T
        return np.apply_along_axis(self.__winner_take_all, 1, projected_data).squeeze()

    def __construct_projection_matrix(self):
        """
        Construct projection matrix from input_dim to hash_dim.
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

    def __winner_take_all(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Winner-take-all operation.

        This also includes quantization step to reduce the number of unique values.
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
