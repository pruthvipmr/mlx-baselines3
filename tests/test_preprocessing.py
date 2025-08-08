"""
Tests for observation preprocessing utilities.

This module tests the preprocessing functions to ensure they correctly handle
different observation space types and perform appropriate transformations.
"""

import pytest
import numpy as np
import gymnasium as gym
import mlx.core as mx

from mlx_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
    maybe_transpose,
    normalize_image,
    preprocess_obs,
    get_obs_shape,
    check_for_nested_spaces,
    flatten_obs,
    get_flattened_obs_dim,
    convert_to_mlx,
)


class TestIsImageSpace:
    """Test image space detection functions."""
    
    def test_box_image_space_channels_last(self):
        """Test image space detection for channels-last format."""
        # Standard image space (H, W, C)
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        assert is_image_space(obs_space) is True
        
        # Grayscale image (H, W)
        obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32), dtype=np.uint8)
        assert is_image_space(obs_space) is True
        
        # Single channel image (H, W, 1)
        obs_space = gym.spaces.Box(low=0, high=255, shape=(28, 28, 1), dtype=np.uint8)
        assert is_image_space(obs_space) is True
        
    def test_box_image_space_channels_first(self):
        """Test image space detection for channels-first format."""
        # RGB image (C, H, W)
        obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        assert is_image_space(obs_space) is True
        
        # Grayscale image (1, H, W)
        obs_space = gym.spaces.Box(low=0, high=255, shape=(1, 32, 32), dtype=np.uint8)
        assert is_image_space(obs_space) is True
        
    def test_non_image_spaces(self):
        """Test that non-image spaces are correctly identified."""
        # Wrong dtype
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(64, 64, 3), dtype=np.float32)
        assert is_image_space(obs_space) is False
        
        # Too many dimensions
        obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64, 2), dtype=np.uint8)
        assert is_image_space(obs_space) is False
        
        # Wrong number of channels
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 7), dtype=np.uint8)
        assert is_image_space(obs_space) is False
        
        # Too small spatial dimensions
        obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 3, 3), dtype=np.uint8)
        assert is_image_space(obs_space) is False
        
        # Vector space
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        assert is_image_space(obs_space) is False
        
        # Discrete space
        obs_space = gym.spaces.Discrete(4)
        assert is_image_space(obs_space) is False
        
    def test_dict_image_space(self):
        """Test image space detection in dictionary observation spaces."""
        obs_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "vector": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        })
        assert is_image_space(obs_space) is True
        
        obs_space = gym.spaces.Dict({
            "vector1": gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "vector2": gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
        })
        assert is_image_space(obs_space) is False
        
    def test_channels_first_detection(self):
        """Test channels-first format detection."""
        # Channels first
        obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        assert is_image_space_channels_first(obs_space) is True
        
        # Channels last
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        assert is_image_space_channels_first(obs_space) is False
        
        # Grayscale
        obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32), dtype=np.uint8)
        assert is_image_space_channels_first(obs_space) is False


class TestMaybeTranspose:
    """Test image transposition functionality."""
    
    def test_channels_last_to_first(self):
        """Test transposing from channels-last to channels-first."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        
        transposed = maybe_transpose(obs, obs_space)
        
        assert transposed.shape == (3, 64, 64)
        # Check that transposition was done correctly
        np.testing.assert_array_equal(transposed[0], obs[:, :, 0])
        np.testing.assert_array_equal(transposed[1], obs[:, :, 1])
        np.testing.assert_array_equal(transposed[2], obs[:, :, 2])
        
    def test_channels_first_unchanged(self):
        """Test that channels-first images are not transposed."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        obs = np.random.randint(0, 256, size=(3, 64, 64), dtype=np.uint8)
        
        result = maybe_transpose(obs, obs_space)
        
        np.testing.assert_array_equal(result, obs)
        
    def test_batch_transpose(self):
        """Test transposing batch of images."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, size=(5, 64, 64, 3), dtype=np.uint8)
        
        transposed = maybe_transpose(obs, obs_space)
        
        assert transposed.shape == (5, 3, 64, 64)
        
    def test_non_image_unchanged(self):
        """Test that non-image observations are unchanged."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        obs = np.random.randn(10).astype(np.float32)
        
        result = maybe_transpose(obs, obs_space)
        
        np.testing.assert_array_equal(result, obs)
        
    def test_grayscale_unchanged(self):
        """Test that grayscale images are unchanged."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32), dtype=np.uint8)
        obs = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        
        result = maybe_transpose(obs, obs_space)
        
        np.testing.assert_array_equal(result, obs)


class TestNormalizeImage:
    """Test image normalization functionality."""
    
    def test_numpy_normalization(self):
        """Test normalizing numpy arrays."""
        obs = np.array([0, 127, 255], dtype=np.uint8)
        normalized = normalize_image(obs)
        
        expected = np.array([0.0, 127/255, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected)
        
    def test_mlx_normalization(self):
        """Test normalizing MLX arrays."""
        obs = mx.array([0, 127, 255])
        normalized = normalize_image(obs)
        
        expected = mx.array([0.0, 127/255, 1.0])
        assert isinstance(normalized, mx.array)
        np.testing.assert_array_almost_equal(np.array(normalized), np.array(expected))
        
    def test_batch_normalization(self):
        """Test normalizing batch of images."""
        obs = np.random.randint(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
        normalized = normalize_image(obs)
        
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)


class TestPreprocessObs:
    """Test the main preprocessing function."""
    
    def test_image_preprocessing(self):
        """Test preprocessing of image observations."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        
        processed = preprocess_obs(obs, obs_space)
        
        # Should be transposed and normalized
        assert processed.shape == (3, 64, 64)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)
        
    def test_image_preprocessing_no_normalize(self):
        """Test preprocessing without normalization."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        
        processed = preprocess_obs(obs, obs_space, normalize_images=False)
        
        # Should be transposed but not normalized
        assert processed.shape == (3, 64, 64)
        assert processed.dtype == np.uint8
        
    def test_image_preprocessing_no_transpose(self):
        """Test preprocessing without transposition."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        
        processed = preprocess_obs(obs, obs_space, transpose_images=False)
        
        # Should be normalized but not transposed
        assert processed.shape == (64, 64, 3)
        assert processed.dtype == np.float32
        
    def test_vector_preprocessing(self):
        """Test preprocessing of vector observations."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        obs = np.random.randn(10).astype(np.float32)
        
        processed = preprocess_obs(obs, obs_space)
        
        # Should be unchanged
        np.testing.assert_array_equal(processed, obs)
        
    def test_dict_preprocessing(self):
        """Test preprocessing of dictionary observations."""
        obs_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
            "vector": gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
        })
        obs = {
            "image": np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8),
            "vector": np.random.randn(5).astype(np.float32),
        }
        
        processed = preprocess_obs(obs, obs_space)
        
        assert isinstance(processed, dict)
        assert "image" in processed
        assert "vector" in processed
        
        # Image should be processed
        assert processed["image"].shape == (3, 32, 32)
        assert processed["image"].dtype == np.float32
        
        # Vector should be unchanged
        np.testing.assert_array_equal(processed["vector"], obs["vector"])
        
    def test_discrete_preprocessing(self):
        """Test preprocessing of discrete observations."""
        obs_space = gym.spaces.Discrete(4)
        obs = 2
        
        processed = preprocess_obs(obs, obs_space)
        
        assert processed == obs


class TestObsShape:
    """Test observation shape utilities."""
    
    def test_box_shape(self):
        """Test getting shape for Box spaces."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(10, 5), dtype=np.float32)
        shape = get_obs_shape(obs_space)
        assert shape == (10, 5)
        
    def test_image_shape_transpose(self):
        """Test getting shape for transposed images."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        shape = get_obs_shape(obs_space)
        # Should account for transposition
        assert shape == (3, 64, 64)
        
    def test_dict_shape(self):
        """Test getting shape for Dict spaces."""
        obs_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(32, 32, 1), dtype=np.uint8),
            "vector": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
        })
        shape = get_obs_shape(obs_space)
        
        assert isinstance(shape, dict)
        assert shape["image"] == (1, 32, 32)  # Transposed
        assert shape["vector"] == (8,)
        
    def test_discrete_shape(self):
        """Test getting shape for Discrete spaces."""
        obs_space = gym.spaces.Discrete(10)
        shape = get_obs_shape(obs_space)
        assert shape == (1,)


class TestNestedSpaces:
    """Test nested space detection."""
    
    def test_dict_not_nested(self):
        """Test that simple dict is not considered nested."""
        obs_space = gym.spaces.Dict({
            "a": gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
            "b": gym.spaces.Discrete(3),
        })
        assert check_for_nested_spaces(obs_space) is False
        
    def test_nested_dict(self):
        """Test detection of nested Dict spaces."""
        obs_space = gym.spaces.Dict({
            "level1": gym.spaces.Dict({
                "level2": gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
            }),
        })
        assert check_for_nested_spaces(obs_space) is True
        
    def test_tuple_space(self):
        """Test detection of Tuple spaces."""
        obs_space = gym.spaces.Tuple((
            gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
            gym.spaces.Discrete(3),
        ))
        assert check_for_nested_spaces(obs_space) is True


class TestFlattenObs:
    """Test observation flattening utilities."""
    
    def test_box_flatten(self):
        """Test flattening Box observations."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(3, 4), dtype=np.float32)
        obs = np.random.randn(3, 4).astype(np.float32)
        
        flattened = flatten_obs(obs, obs_space)
        assert flattened.shape == (12,)
        
    def test_dict_flatten(self):
        """Test flattening Dict observations."""
        obs_space = gym.spaces.Dict({
            "a": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "b": gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        })
        obs = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 4.0, 5.0]),
        }
        
        flattened = flatten_obs(obs, obs_space)
        # Keys are sorted, so "a" comes before "b"
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(flattened, expected)
        
    def test_discrete_flatten(self):
        """Test flattening discrete observations."""
        obs_space = gym.spaces.Discrete(5)
        obs = 3
        
        flattened = flatten_obs(obs, obs_space)
        assert flattened.shape == (1,)
        assert flattened[0] == 3
        
    def test_flattened_dim(self):
        """Test getting flattened dimension."""
        obs_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
            "vector": gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
        })
        
        dim = get_flattened_obs_dim(obs_space)
        expected = 32 * 32 * 3 + 5  # Image pixels + vector elements
        assert dim == expected


class TestConvertToMLX:
    """Test MLX conversion functionality."""
    
    def test_numpy_to_mlx(self):
        """Test converting numpy arrays to MLX."""
        obs = np.random.randn(5, 3).astype(np.float32)
        mlx_obs = convert_to_mlx(obs)
        
        assert isinstance(mlx_obs, mx.array)
        np.testing.assert_array_equal(np.array(mlx_obs), obs)
        
    def test_dict_to_mlx(self):
        """Test converting dict of numpy arrays to MLX."""
        obs = {
            "a": np.random.randn(3).astype(np.float32),
            "b": np.random.randn(2, 4).astype(np.float32),
        }
        mlx_obs = convert_to_mlx(obs)
        
        assert isinstance(mlx_obs, dict)
        assert isinstance(mlx_obs["a"], mx.array)
        assert isinstance(mlx_obs["b"], mx.array)
        np.testing.assert_array_equal(np.array(mlx_obs["a"]), obs["a"])
        np.testing.assert_array_equal(np.array(mlx_obs["b"]), obs["b"])


if __name__ == "__main__":
    pytest.main([__file__])
