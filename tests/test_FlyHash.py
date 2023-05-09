import numpy as np
import pytest

from flyhash.FlyHash import FlyHash

__author__ = "TeddyHuang-00"
__copyright__ = "TeddyHuang-00"
__license__ = "MIT"


def test_flyhash():
    normal_input_dim = 10
    normal_hash_dim = 100
    normal_int_density = 6
    normal_float_density = 0.05
    normal_sparsity = 0.1
    normal_quant_step = 0.1
    normal_dtype = np.int16
    normal_seed = 0

    wrong_input_dim = 0
    wrong_hash_dim = 0
    wrong_int_density = 0
    wrong_float_density = 1.0
    wrong_sparsity = 0.0
    wrong_quant_step = 0.0
    wrong_dtype = np.float32

    # Test weird input values
    with pytest.raises(AssertionError):
        FlyHash(
            wrong_input_dim,
            normal_hash_dim,
            normal_int_density,
            normal_sparsity,
            normal_quant_step,
            normal_dtype,
        )
    with pytest.raises(AssertionError):
        FlyHash(
            normal_input_dim,
            wrong_hash_dim,
            normal_int_density,
            normal_sparsity,
            normal_quant_step,
            normal_dtype,
        )
    with pytest.raises(AssertionError):
        FlyHash(
            normal_input_dim,
            normal_hash_dim,
            wrong_int_density,
            normal_sparsity,
            normal_quant_step,
            normal_dtype,
        )
    with pytest.raises(AssertionError):
        FlyHash(
            normal_input_dim,
            normal_hash_dim,
            wrong_float_density,
            normal_sparsity,
            normal_quant_step,
            normal_dtype,
        )
    with pytest.raises(AssertionError):
        FlyHash(
            normal_input_dim,
            normal_hash_dim,
            normal_int_density,
            wrong_sparsity,
            normal_quant_step,
            normal_dtype,
        )
    with pytest.raises(AssertionError):
        FlyHash(
            normal_input_dim,
            normal_hash_dim,
            normal_int_density,
            normal_sparsity,
            wrong_quant_step,
            normal_dtype,
        )
    with pytest.raises(AssertionError):
        FlyHash(
            normal_input_dim,
            normal_hash_dim,
            normal_int_density,
            normal_sparsity,
            normal_quant_step,
            wrong_dtype,
        )

    fixed_hasher = FlyHash(
        normal_input_dim,
        normal_hash_dim,
        normal_int_density,
        normal_sparsity,
        dtype=normal_dtype,
    )
    random_hasher = FlyHash(
        normal_input_dim,
        normal_hash_dim,
        normal_float_density,
        normal_sparsity,
        dtype=normal_dtype,
    )

    normal_matrix_data = np.random.randn(10, normal_input_dim)
    normal_vector_data = np.random.randn(normal_input_dim)
    wrong_matrix_data = np.random.randn(10, normal_input_dim + 1)
    wrong_vector_data = np.random.randn(normal_input_dim + 1)
    wrong_dimension_data = np.random.randn(2, 5, normal_input_dim)

    # Test weird input_data values
    for hasher in (fixed_hasher, random_hasher):
        for input_data in (wrong_matrix_data, wrong_vector_data, wrong_dimension_data):
            with pytest.raises(AssertionError):
                hasher(input_data)

    fixed_matrix = fixed_hasher(normal_matrix_data)
    fixed_vector = fixed_hasher(normal_vector_data)
    random_matrix = random_hasher(normal_matrix_data)
    random_vector = random_hasher(normal_vector_data)

    assert fixed_matrix.shape == (10, normal_hash_dim)
    assert fixed_vector.shape == (normal_hash_dim,)
    assert fixed_matrix.dtype == normal_dtype
    assert random_matrix.shape == (10, normal_hash_dim)
    assert random_vector.shape == (normal_hash_dim,)
    assert random_matrix.dtype == normal_dtype

    # Test random seed
    hasher_1 = FlyHash(normal_input_dim, normal_hash_dim, seed=normal_seed)
    hasher_2 = FlyHash(normal_input_dim, normal_hash_dim, seed=normal_seed)
    assert np.all(hasher_1(normal_matrix_data) == hasher_2(normal_matrix_data))
    assert np.all(hasher_1.projection_matrix == hasher_2.projection_matrix)

    # Test clipping
    binary_clip_hasher = FlyHash(
        normal_input_dim,
        normal_hash_dim,
        sparsity=normal_sparsity,
        quant_step=None,
        dtype=normal_dtype,
        seed=normal_seed,
    )
    binary_matrix = binary_clip_hasher(normal_matrix_data)
    binary_vector = binary_clip_hasher(normal_vector_data)

    assert binary_matrix.max() == 1
    assert binary_matrix.min() == 0
    count = (binary_vector > 0).sum()
    assert np.allclose(count / normal_hash_dim, normal_sparsity, atol=1e-2)

    overflow_clip_hasher = FlyHash(
        normal_input_dim,
        normal_hash_dim,
        sparsity=normal_sparsity,
        quant_step=1e-15,
        dtype=normal_dtype,
        seed=normal_seed,
    )
    overflow_matrix = overflow_clip_hasher(normal_matrix_data)
    overflow_vector = overflow_clip_hasher(normal_vector_data)
    assert overflow_matrix.max() == np.iinfo(normal_dtype).max
    assert overflow_matrix.min() == 0
    count = (overflow_vector > 0).sum()
    assert np.allclose(count / normal_hash_dim, normal_sparsity, atol=1e-2)
