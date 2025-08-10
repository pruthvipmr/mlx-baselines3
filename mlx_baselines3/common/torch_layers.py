"""
MLX Neural Network Layers

MLX-based replacements for PyTorch neural network components.
Provides base classes and utilities for building neural networks in MLX.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, Callable
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_baselines3.common.type_aliases import MlxArray


class MlxModule(ABC):
    """Base class for MLX neural network modules."""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        
    @abstractmethod
    def __call__(self, x: MlxArray) -> MlxArray:
        """Forward pass through the module."""
        pass
    
    def parameters(self) -> Dict[str, MlxArray]:
        """Get all parameters of this module and its submodules."""
        params = {}
        
        # Add direct parameters
        for name, param in self._parameters.items():
            params[name] = param
            
        # Add parameters from submodules
        for module_name, module in self._modules.items():
            if hasattr(module, 'parameters'):
                module_params = module.parameters()
                for param_name, param in module_params.items():
                    params[f"{module_name}.{param_name}"] = param
        
        return params
    
    def named_parameters(self) -> Dict[str, MlxArray]:
        """Get named parameters for this module."""
        return self.parameters()
    
    def add_parameter(self, name: str, param: MlxArray) -> None:
        """Add a parameter to this module."""
        self._parameters[name] = param
    
    def add_module(self, name: str, module: 'MlxModule') -> None:
        """Add a submodule to this module."""
        self._modules[name] = module
        setattr(self, name, module)
    
    def state_dict(self) -> Dict[str, MlxArray]:
        """
        Get state dictionary containing all parameters and buffers.
        
        Returns:
            Dictionary mapping parameter names to their values
        """
        return self.parameters()
    
    def load_state_dict(self, state_dict: Dict[str, MlxArray], strict: bool = True) -> None:
        """
        Load parameters from a state dictionary.
        
        Args:
            state_dict: Dictionary containing parameter names and values
            strict: Whether to strictly enforce that the keys match exactly
            
        Raises:
            KeyError: If strict=True and keys don't match exactly
            ValueError: If parameter shapes don't match
        """
        current_params = self.parameters()
        
        if strict:
            # Check for missing and unexpected keys
            missing_keys = set(current_params.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(current_params.keys())
            
            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")
        
        # Load parameters, checking shapes
        for name, param in state_dict.items():
            if name in current_params:
                current_param = current_params[name]
                if current_param.shape != param.shape:
                    raise ValueError(
                        f"Parameter '{name}' shape mismatch: "
                        f"expected {current_param.shape}, got {param.shape}"
                    )
                
                # Update the parameter in the appropriate location
                self._update_parameter(name, param)
            elif not strict:
                # Warn about skipped keys if not in strict mode
                print(f"Warning: Skipping unexpected key '{name}' in state_dict")
    
    def _update_parameter(self, name: str, param: MlxArray) -> None:
        """
        Update a parameter by name, handling nested modules.
        
        Args:
            name: Parameter name (may contain dots for nested modules)
            param: New parameter value
        """
        if "." in name:
            # Handle nested module parameters
            module_name, rest = name.split(".", 1)
            if module_name in self._modules:
                self._modules[module_name]._update_parameter(rest, param)
            else:
                raise KeyError(f"Module '{module_name}' not found")
        else:
            # Direct parameter
            if name in self._parameters:
                self._parameters[name] = param
            else:
                raise KeyError(f"Parameter '{name}' not found")


class MlxSequential(MlxModule):
    """Sequential container for MLX layers."""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        
        for i, layer in enumerate(layers):
            self.layers.append(layer)
            self.add_module(f"layer_{i}", layer)
    
    def __call__(self, x: MlxArray) -> MlxArray:
        """Forward pass through all layers sequentially."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def append(self, layer: MlxModule) -> None:
        """Add a layer to the end of the sequence."""
        idx = len(self.layers)
        self.layers.append(layer)
        self.add_module(f"layer_{idx}", layer)


class MlxLinear(MlxModule):
    """MLX linear layer."""
    
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias
        
        # Initialize weights with Xavier uniform
        weight = mx.random.uniform(
            low=-mx.sqrt(6.0 / (input_dim + output_dim)),
            high=mx.sqrt(6.0 / (input_dim + output_dim)),
            shape=(output_dim, input_dim)
        )
        self.add_parameter("weight", weight)
        
        if bias:
            bias_param = mx.zeros((output_dim,))
            self.add_parameter("bias", bias_param)
    
    def __call__(self, x: MlxArray) -> MlxArray:
        """Forward pass through linear layer."""
        output = mx.matmul(x, self._parameters["weight"].T)
        if self.use_bias:
            output = output + self._parameters["bias"]
        return output


class MlxActivation(MlxModule):
    """Wrapper for MLX activation functions."""
    
    def __init__(self, activation_fn: Callable[[MlxArray], MlxArray]):
        super().__init__()
        self.activation_fn = activation_fn
    
    def __call__(self, x: MlxArray) -> MlxArray:
        """Apply activation function."""
        return self.activation_fn(x)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> MlxSequential:
    """
    Create a multi-layer perceptron (MLP) with specified architecture.
    
    Args:
        input_dim: Dimension of the input
        output_dim: Dimension of the output
        net_arch: Architecture of the neural net. List of hidden layer sizes
        activation_fn: Activation function to use after each hidden layer
        squash_output: Whether to squash the output using a Tanh activation
        with_bias: Whether to include bias terms in linear layers
        
    Returns:
        MlxSequential network
    """
    if len(net_arch) > 0:
        modules = []
        
        # Input layer
        modules.append(MlxLinear(input_dim, net_arch[0], bias=with_bias))
        modules.append(_get_mlx_activation(activation_fn))
        
        # Hidden layers
        for i in range(len(net_arch) - 1):
            modules.append(MlxLinear(net_arch[i], net_arch[i + 1], bias=with_bias))
            modules.append(_get_mlx_activation(activation_fn))
        
        # Output layer
        modules.append(MlxLinear(net_arch[-1], output_dim, bias=with_bias))
    else:
        # No hidden layers
        modules = [MlxLinear(input_dim, output_dim, bias=with_bias)]
    
    # Add output squashing if requested
    if squash_output:
        modules.append(_get_mlx_activation(nn.Tanh))
    
    return MlxSequential(*modules)


def _get_mlx_activation(activation_fn: Type[nn.Module]) -> MlxActivation:
    """Get MLX activation function wrapper."""
    if activation_fn == nn.ReLU:
        return MlxActivation(nn.relu)
    elif activation_fn == nn.Tanh:
        return MlxActivation(nn.tanh)
    elif activation_fn == nn.Sigmoid:
        return MlxActivation(nn.sigmoid)
    elif activation_fn == nn.LeakyReLU:
        return MlxActivation(lambda x: nn.leaky_relu(x, negative_slope=0.01))
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")


def init_weights(module: MlxModule, gain: float = 1.0, init_type: str = "xavier") -> None:
    """
    Initialize weights of an MLX module.
    
    Args:
        module: MLX module to initialize
        gain: Gain factor for initialization
        init_type: Type of initialization ('xavier', 'orthogonal', 'normal')
    """
    parameters = module.parameters()
    
    for name, param in parameters.items():
        if "weight" in name and len(param.shape) >= 2:
            if init_type == "xavier":
                fan_in, fan_out = param.shape[-2], param.shape[-1]
                std = gain * mx.sqrt(2.0 / (fan_in + fan_out))
                new_param = mx.random.normal(param.shape) * std
            elif init_type == "orthogonal":
                # Simplified orthogonal initialization
                new_param = mx.random.normal(param.shape) * gain / mx.sqrt(float(param.shape[-2]))
            elif init_type == "normal":
                new_param = mx.random.normal(param.shape) * gain * 0.01
            else:
                raise ValueError(f"Unknown initialization type: {init_type}")
            
            # Update parameter in-place
            module._parameters[name.split(".")[-1]] = new_param
        elif "bias" in name:
            # Initialize biases to zero
            module._parameters[name.split(".")[-1]] = mx.zeros(param.shape)


class BaseFeaturesExtractor(MlxModule):
    """
    Base class for feature extractors.
    
    Feature extractors are used to process observations before feeding them
    to the policy or value function networks.
    """
    
    def __init__(self, observation_space, features_dim: int = 0):
        super().__init__()
        self.observation_space = observation_space
        self._features_dim = features_dim
    
    @property
    def features_dim(self) -> int:
        """Return the dimension of extracted features."""
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that flattens the observation.
    
    This is used as a default feature extractor for vector observations.
    """
    
    def __init__(self, observation_space):
        # Calculate flattened dimension
        if hasattr(observation_space, 'shape'):
            features_dim = int(mx.prod(mx.array(observation_space.shape)))
        else:
            # For discrete spaces
            features_dim = 1
            
        super().__init__(observation_space, features_dim)
    
    def __call__(self, observations: MlxArray) -> MlxArray:
        """Flatten observations."""
        # If observations are already 1D (batch dimension + features), keep as is
        if len(observations.shape) == 2:
            return observations
        # Otherwise flatten everything except batch dimension
        return mx.reshape(observations, (observations.shape[0], -1))


class MlpExtractor(BaseFeaturesExtractor):
    """
    MLP feature extractor for vector observations.
    
    Applies an MLP to extract features from vector observations.
    """
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(observation_space, features_dim)
        
        if net_arch is None:
            net_arch = [features_dim]
        
        if hasattr(observation_space, 'shape') and observation_space.shape is not None:
            input_dim = int(mx.prod(mx.array(observation_space.shape)))
        else:
            # For Dict observation spaces, flatten each component
            from mlx_baselines3.common.preprocessing import get_flattened_obs_dim
            input_dim = get_flattened_obs_dim(observation_space)
        
        self.mlp = create_mlp(
            input_dim=input_dim,
            output_dim=features_dim,
            net_arch=net_arch[:-1],  # Exclude last layer (handled by output_dim)
            activation_fn=activation_fn,
        )
        # Register the MLP as a submodule for proper parameter discovery
        self.add_module("mlp", self.mlp)
    
    def __call__(self, observations: MlxArray) -> MlxArray:
        """Extract features using MLP."""
        # Flatten observations if needed
        if len(observations.shape) > 2:
            observations = mx.reshape(observations, (observations.shape[0], -1))
        
        return self.mlp(observations)
