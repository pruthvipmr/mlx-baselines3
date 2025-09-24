"""
Tests for MLX neural network layers.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import gymnasium as gym

from mlx_baselines3.common.torch_layers import (
    MlxModule,
    MlxSequential,
    MlxLinear,
    MlxActivation,
    create_mlp,
    init_weights,
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)


class TestMlxModule:
    """Test MlxModule base class."""

    def test_abstract_module(self):
        """Test that MlxModule is abstract."""
        with pytest.raises(TypeError):
            MlxModule()


class TestMlxLinear:
    """Test MlxLinear layer."""

    def test_initialization(self):
        """Test linear layer initialization."""
        layer = MlxLinear(10, 5, bias=True)

        assert layer.input_dim == 10
        assert layer.output_dim == 5
        assert layer.use_bias is True

        params = layer.parameters()
        assert "weight" in params
        assert "bias" in params
        assert params["weight"].shape == (5, 10)
        assert params["bias"].shape == (5,)

    def test_initialization_no_bias(self):
        """Test linear layer initialization without bias."""
        layer = MlxLinear(10, 5, bias=False)

        assert layer.use_bias is False

        params = layer.parameters()
        assert "weight" in params
        assert "bias" not in params

    def test_forward_pass(self):
        """Test forward pass through linear layer."""
        layer = MlxLinear(3, 2, bias=True)
        x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        output = layer(x)

        assert output.shape == (2, 2)
        assert output.dtype == mx.float32

    def test_forward_pass_no_bias(self):
        """Test forward pass through linear layer without bias."""
        layer = MlxLinear(3, 2, bias=False)
        x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        output = layer(x)

        assert output.shape == (2, 2)
        assert output.dtype == mx.float32


class TestMlxActivation:
    """Test MlxActivation wrapper."""

    def test_relu_activation(self):
        """Test ReLU activation."""
        activation = MlxActivation(nn.relu)
        x = mx.array([[-1.0, 0.0, 1.0, 2.0]])

        output = activation(x)
        expected = mx.array([[0.0, 0.0, 1.0, 2.0]])

        assert mx.allclose(output, expected)

    def test_tanh_activation(self):
        """Test Tanh activation."""
        activation = MlxActivation(nn.tanh)
        x = mx.array([[0.0, 1.0, -1.0]])

        output = activation(x)

        assert output.shape == (1, 3)
        assert mx.allclose(output[0, 0], mx.array(0.0), atol=1e-6)


class TestMlxSequential:
    """Test MlxSequential container."""

    def test_initialization(self):
        """Test sequential container initialization."""
        layer1 = MlxLinear(10, 5)
        layer2 = MlxActivation(nn.relu)
        layer3 = MlxLinear(5, 2)

        sequential = MlxSequential(layer1, layer2, layer3)

        assert len(sequential.layers) == 3
        assert hasattr(sequential, "layer_0")
        assert hasattr(sequential, "layer_1")
        assert hasattr(sequential, "layer_2")

    def test_forward_pass(self):
        """Test forward pass through sequential container."""
        layer1 = MlxLinear(3, 2)
        layer2 = MlxActivation(nn.relu)

        sequential = MlxSequential(layer1, layer2)
        x = mx.array([[1.0, 2.0, 3.0]])

        output = sequential(x)

        assert output.shape == (1, 2)
        assert mx.all(output >= 0)  # ReLU should make all values non-negative

    def test_append_layer(self):
        """Test appending layers to sequential container."""
        layer1 = MlxLinear(3, 2)
        sequential = MlxSequential(layer1)

        assert len(sequential.layers) == 1

        layer2 = MlxActivation(nn.relu)
        sequential.append(layer2)

        assert len(sequential.layers) == 2
        assert hasattr(sequential, "layer_1")

    def test_parameter_collection(self):
        """Test parameter collection from sequential container."""
        layer1 = MlxLinear(3, 2, bias=True)
        layer2 = MlxLinear(2, 1, bias=False)

        sequential = MlxSequential(layer1, layer2)
        params = sequential.parameters()

        # Should have weight and bias from layer1, weight from layer2
        expected_keys = {"layer_0.weight", "layer_0.bias", "layer_1.weight"}
        assert set(params.keys()) == expected_keys


class TestCreateMlp:
    """Test MLP creation utility."""

    def test_basic_mlp(self):
        """Test basic MLP creation."""
        mlp = create_mlp(
            input_dim=10,
            output_dim=2,
            net_arch=[64, 32],
            activation_fn=nn.ReLU,
        )

        x = mx.random.normal((5, 10))
        output = mlp(x)

        assert output.shape == (5, 2)

    def test_mlp_no_hidden_layers(self):
        """Test MLP with no hidden layers."""
        mlp = create_mlp(
            input_dim=10,
            output_dim=2,
            net_arch=[],
            activation_fn=nn.ReLU,
        )

        x = mx.random.normal((5, 10))
        output = mlp(x)

        assert output.shape == (5, 2)
        assert len(mlp.layers) == 1  # Only output layer

    def test_mlp_with_squash_output(self):
        """Test MLP with squashed output."""
        mlp = create_mlp(
            input_dim=10,
            output_dim=2,
            net_arch=[32],
            activation_fn=nn.ReLU,
            squash_output=True,
        )

        x = mx.random.normal((5, 10))
        output = mlp(x)

        assert output.shape == (5, 2)
        # Output should be in [-1, 1] due to tanh
        assert mx.all(output >= -1.0)
        assert mx.all(output <= 1.0)

    def test_mlp_no_bias(self):
        """Test MLP creation without bias."""
        mlp = create_mlp(
            input_dim=10,
            output_dim=2,
            net_arch=[32],
            activation_fn=nn.ReLU,
            with_bias=False,
        )

        params = mlp.parameters()
        bias_keys = [key for key in params.keys() if "bias" in key]
        assert len(bias_keys) == 0


class TestInitWeights:
    """Test weight initialization."""

    def test_xavier_initialization(self):
        """Test Xavier initialization."""
        layer = MlxLinear(10, 5)
        original_weight = mx.array(layer.parameters()["weight"])

        init_weights(layer, gain=1.0, init_type="xavier")
        new_weight = layer.parameters()["weight"]

        # Weights should have changed
        assert not mx.allclose(original_weight, new_weight)

        # Check if variance is approximately correct for Xavier init
        variance = mx.var(new_weight)
        expected_variance = 2.0 / (10 + 5)  # Xavier formula
        assert abs(variance - expected_variance) < 0.1

    def test_orthogonal_initialization(self):
        """Test orthogonal initialization."""
        layer = MlxLinear(10, 5)
        original_weight = mx.array(layer.parameters()["weight"])

        init_weights(layer, gain=1.0, init_type="orthogonal")
        new_weight = layer.parameters()["weight"]

        # Weights should have changed
        assert not mx.allclose(original_weight, new_weight)

    def test_normal_initialization(self):
        """Test normal initialization."""
        layer = MlxLinear(10, 5)
        original_weight = mx.array(layer.parameters()["weight"])

        init_weights(layer, gain=1.0, init_type="normal")
        new_weight = layer.parameters()["weight"]

        # Weights should have changed
        assert not mx.allclose(original_weight, new_weight)

    def test_bias_initialization(self):
        """Test that biases are initialized to zero."""
        layer = MlxLinear(10, 5, bias=True)

        init_weights(layer, gain=1.0, init_type="xavier")
        bias = layer.parameters()["bias"]

        assert mx.allclose(bias, mx.zeros(5))


class TestBaseFeaturesExtractor:
    """Test BaseFeaturesExtractor."""

    def test_abstract_extractor(self):
        """Test that BaseFeaturesExtractor is abstract."""
        observation_space = gym.spaces.Box(-1, 1, (4,))

        # Should raise error for abstract class instantiation
        with pytest.raises(TypeError):
            BaseFeaturesExtractor(observation_space, features_dim=10)


class TestFlattenExtractor:
    """Test FlattenExtractor."""

    def test_box_space_features_dim(self):
        """Test features dimension calculation for Box space."""
        observation_space = gym.spaces.Box(-1, 1, (4, 3))
        extractor = FlattenExtractor(observation_space)

        assert extractor.features_dim == 12  # 4 * 3

    def test_discrete_space_features_dim(self):
        """Test features dimension for discrete space."""
        observation_space = gym.spaces.Discrete(5)
        extractor = FlattenExtractor(observation_space)

        assert extractor.features_dim == 1

    def test_flatten_2d_observations(self):
        """Test flattening 2D observations (batch + features)."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        extractor = FlattenExtractor(observation_space)

        obs = mx.random.normal((3, 4))  # batch_size=3, features=4
        features = extractor(obs)

        assert features.shape == (3, 4)
        assert mx.array_equal(features, obs)

    def test_flatten_3d_observations(self):
        """Test flattening 3D observations."""
        observation_space = gym.spaces.Box(-1, 1, (2, 3))
        extractor = FlattenExtractor(observation_space)

        obs = mx.random.normal((5, 2, 3))  # batch_size=5, shape=(2,3)
        features = extractor(obs)

        assert features.shape == (5, 6)  # 2 * 3 = 6


class TestMlpExtractor:
    """Test MlpExtractor."""

    def test_initialization(self):
        """Test MLP extractor initialization."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        extractor = MlpExtractor(
            observation_space,
            features_dim=32,
            net_arch=[64],
            activation_fn=nn.ReLU,
        )

        assert extractor.features_dim == 32

    def test_feature_extraction(self):
        """Test feature extraction with MLP."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        extractor = MlpExtractor(
            observation_space,
            features_dim=16,
            net_arch=[32],
            activation_fn=nn.ReLU,
        )

        obs = mx.random.normal((3, 4))
        features = extractor(obs)

        assert features.shape == (3, 16)

    def test_multidimensional_observations(self):
        """Test MLP extractor with multidimensional observations."""
        observation_space = gym.spaces.Box(-1, 1, (2, 3))
        extractor = MlpExtractor(
            observation_space,
            features_dim=8,
            net_arch=[16],
            activation_fn=nn.ReLU,
        )

        obs = mx.random.normal((5, 2, 3))
        features = extractor(obs)

        assert features.shape == (5, 8)
