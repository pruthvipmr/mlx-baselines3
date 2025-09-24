"""
MLX-native optimizer adapters for stable, functional gradient-based
optimization.

This module provides optimizer adapters that work with MLX's functional
autograd system, maintaining optimizer state (like Adam moments) separately
from model parameters.
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple, TypedDict, Union

import mlx.core as mx


class OptimizerState(TypedDict):
    """Type definition for optimizer state dictionaries."""

    step: int
    # Adam-specific state (will be empty for SGD)
    m: Dict[str, mx.array]  # First moment estimates
    v: Dict[str, mx.array]  # Second moment estimates


class AdamAdapter:
    """
    Functional Adam optimizer adapter for MLX.

    Maintains optimizer state externally and provides functional updates
    that return new parameters and state without mutating inputs.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize Adam optimizer adapter.

        Args:
            learning_rate: Learning rate (constant or schedule function)
            betas: Adam beta parameters (beta1, beta2)
            eps: Epsilon for numerical stability
            weight_decay: L2 weight decay coefficient
        """
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self._manual_lr: Optional[float] = None

    def init_state(self, params: Dict[str, mx.array]) -> OptimizerState:
        """
        Initialize optimizer state for given parameters.

        Args:
            params: Dictionary of parameter arrays

        Returns:
            Initial optimizer state with zero moments
        """
        # Initialize moment estimates with zeros, matching parameter shapes
        m = {k: mx.zeros_like(v) for k, v in params.items()}
        v = {k: mx.zeros_like(v) for k, v in params.items()}

        return OptimizerState(step=0, m=m, v=v)

    def update(
        self,
        params: Dict[str, mx.array],
        grads: Dict[str, mx.array],
        state: OptimizerState,
    ) -> Tuple[Dict[str, mx.array], OptimizerState]:
        """
        Perform Adam update step.

        Args:
            params: Current parameter dictionary
            grads: Gradient dictionary (same keys as params)
            state: Current optimizer state

        Returns:
            Tuple of (updated_params, updated_state)
        """
        # Get current learning rate (support schedules)
        if self._manual_lr is not None:
            current_lr = self._manual_lr
        elif callable(self.learning_rate):
            current_lr = float(self.learning_rate(state["step"]))
        else:
            current_lr = float(self.learning_rate)

        # Increment step count
        new_step = state["step"] + 1

        # Bias correction terms
        bias_correction1 = 1 - self.beta1**new_step
        bias_correction2 = 1 - self.beta2**new_step

        # Update moment estimates and parameters
        new_m = {}
        new_v = {}
        new_params = {}

        for key in params.keys():
            if key not in grads:
                # No gradient for this parameter, keep unchanged
                new_params[key] = params[key]
                new_m[key] = state["m"][key]
                new_v[key] = state["v"][key]
                continue

            param = params[key]
            grad = grads[key]

            # Apply weight decay if specified
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update biased first moment estimate
            new_m[key] = self.beta1 * state["m"][key] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            new_v[key] = self.beta2 * state["v"][key] + (1 - self.beta2) * (grad**2)

            # Bias-corrected moment estimates
            m_hat = new_m[key] / bias_correction1
            v_hat = new_v[key] / bias_correction2

            # Update parameters
            new_params[key] = param - current_lr * m_hat / (mx.sqrt(v_hat) + self.eps)

        # Create new state
        new_state = OptimizerState(step=new_step, m=new_m, v=new_v)

        return new_params, new_state

    def set_current_learning_rate(self, learning_rate: float) -> None:
        """Manually override the learning rate used during updates."""
        self._manual_lr = float(learning_rate)


class RMSPropAdapter:
    """
    Functional RMSProp optimizer adapter for MLX.

    RMSProp maintains a moving average of squared gradients to adapt
    the learning rate per parameter, which is effective for non-stationary objectives.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize RMSProp optimizer adapter.

        Args:
            learning_rate: Learning rate (constant or schedule function)
            alpha: Smoothing constant (decay factor for moving average)
            eps: Epsilon for numerical stability
            weight_decay: L2 weight decay coefficient
        """
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self._manual_lr: Optional[float] = None

    def init_state(self, params: Dict[str, mx.array]) -> OptimizerState:
        """
        Initialize optimizer state for RMSProp.

        Args:
            params: Dictionary of parameter arrays

        Returns:
            Initial optimizer state with zero squared gradient averages
        """
        # Initialize squared gradient moving averages
        v = {k: mx.zeros_like(param) for k, param in params.items()}

        return OptimizerState(
            step=0,
            m={},  # RMSProp doesn't use first moments
            v=v,
        )

    def update(
        self,
        params: Dict[str, mx.array],
        grads: Dict[str, mx.array],
        state: OptimizerState,
    ) -> Tuple[Dict[str, mx.array], OptimizerState]:
        """
        Perform RMSProp update step.

        Args:
            params: Current parameter dictionary
            grads: Gradient dictionary
            state: Current optimizer state

        Returns:
            Tuple of (updated_params, updated_state)
        """
        # Get current learning rate
        if self._manual_lr is not None:
            current_lr = self._manual_lr
        elif callable(self.learning_rate):
            current_lr = float(self.learning_rate(state["step"]))
        else:
            current_lr = float(self.learning_rate)

        new_step = state["step"] + 1
        new_params = {}
        new_v = {}

        for key in params.keys():
            if key not in grads:
                new_params[key] = params[key]
                new_v[key] = state["v"][key]
                continue

            param = params[key]
            grad = grads[key]

            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update squared gradient moving average
            new_v[key] = self.alpha * state["v"][key] + (1 - self.alpha) * (grad**2)

            # Update parameters
            new_params[key] = param - current_lr * grad / (
                mx.sqrt(new_v[key]) + self.eps
            )

        new_state = OptimizerState(step=new_step, m={}, v=new_v)

        return new_params, new_state

    def set_current_learning_rate(self, learning_rate: float) -> None:
        """Manually override the learning rate used during updates."""
        self._manual_lr = float(learning_rate)


class SGDAdapter:
    """
    Functional SGD optimizer adapter for MLX.

    Simple SGD implementation for testing and fallback purposes.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """
        Initialize SGD optimizer adapter.

        Args:
            learning_rate: Learning rate (constant or schedule function)
            momentum: Momentum coefficient
            weight_decay: L2 weight decay coefficient
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._manual_lr: Optional[float] = None

    def init_state(self, params: Dict[str, mx.array]) -> OptimizerState:
        """
        Initialize optimizer state for SGD.

        Args:
            params: Dictionary of parameter arrays

        Returns:
            Initial optimizer state
        """
        # For SGD with momentum, we track velocity
        if self.momentum > 0:
            m = {k: mx.zeros_like(v) for k, v in params.items()}
        else:
            m = {}

        return OptimizerState(
            step=0,
            m=m,
            v={},  # SGD doesn't use second moments
        )

    def update(
        self,
        params: Dict[str, mx.array],
        grads: Dict[str, mx.array],
        state: OptimizerState,
    ) -> Tuple[Dict[str, mx.array], OptimizerState]:
        """
        Perform SGD update step.

        Args:
            params: Current parameter dictionary
            grads: Gradient dictionary
            state: Current optimizer state

        Returns:
            Tuple of (updated_params, updated_state)
        """
        # Get current learning rate
        if self._manual_lr is not None:
            current_lr = self._manual_lr
        elif callable(self.learning_rate):
            current_lr = float(self.learning_rate(state["step"]))
        else:
            current_lr = float(self.learning_rate)

        new_step = state["step"] + 1
        new_params = {}
        new_m = {}

        for key in params.keys():
            if key not in grads:
                new_params[key] = params[key]
                if self.momentum > 0:
                    new_m[key] = state["m"][key]
                continue

            param = params[key]
            grad = grads[key]

            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            if self.momentum > 0:
                # Update momentum
                new_m[key] = self.momentum * state["m"][key] + grad
                # Update parameters using momentum
                new_params[key] = param - current_lr * new_m[key]
            else:
                # Simple SGD update
                new_params[key] = param - current_lr * grad
                new_m[key] = mx.zeros_like(param)  # Dummy for consistency

        new_state = OptimizerState(step=new_step, m=new_m, v={})

        return new_params, new_state

    def set_current_learning_rate(self, learning_rate: float) -> None:
        """Manually override the learning rate used during updates."""
        self._manual_lr = float(learning_rate)


def create_optimizer_adapter(
    optimizer_name: str,
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    **kwargs: Any,
) -> Union[AdamAdapter, RMSPropAdapter, SGDAdapter]:
    """
    Factory function to create optimizer adapters.

    Args:
        optimizer_name: Name of optimizer ('adam', 'rmsprop', 'sgd')
        learning_rate: Learning rate (constant or callable schedule)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer adapter instance

    Raises:
        ValueError: If optimizer_name is not supported
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return AdamAdapter(learning_rate=learning_rate, **kwargs)
    elif optimizer_name == "rmsprop":
        return RMSPropAdapter(learning_rate=learning_rate, **kwargs)
    elif optimizer_name == "sgd":
        return SGDAdapter(learning_rate=learning_rate, **kwargs)
    else:
        supported = ("adam", "rmsprop", "sgd")
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. Supported: {supported}"
        )


def clip_grad_norm(
    grads: Dict[str, Optional[mx.array]], max_norm: float
) -> Tuple[Dict[str, Optional[mx.array]], float]:
    """
    Clip gradients by global norm.

    Args:
        grads: Dictionary of gradients
        max_norm: Maximum allowed gradient norm

    Returns:
        Tuple of (clipped_gradients, original_norm)
    """
    # Compute total gradient norm
    total_norm_squared = 0.0
    for grad in grads.values():
        if grad is not None:
            total_norm_squared += float(mx.sum(grad**2))
    total_norm = math.sqrt(total_norm_squared)

    # Clip gradients if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-8)
        clipped_grads: Dict[str, Optional[mx.array]] = {
            key: grad * clip_coef if grad is not None else None
            for key, grad in grads.items()
        }
        return clipped_grads, total_norm
    return grads, total_norm


def compute_loss_and_grads(
    loss_fn: Callable[[Dict[str, mx.array]], mx.array], params: Dict[str, mx.array]
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Centralized gradient computation helper.

    Computes loss and gradients using MLX's value_and_grad in a standardized way.

    Args:
        loss_fn: Pure function that takes params dict and returns scalar loss
        params: Dictionary of parameters

    Returns:
        Tuple of (loss_value, gradients_dict)
    """
    # Create the value_and_grad function
    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    # Compute loss and gradients
    loss_value, grads = loss_and_grad_fn(params)

    return loss_value, grads
