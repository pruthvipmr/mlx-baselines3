"""
Tests for vectorized environments.

This module tests the vectorized environment implementations to ensure
they work correctly with gym environments and maintain API compatibility.
"""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock

from mlx_baselines3.common.vec_env import (
    VecEnv,
    VecEnvWrapper,
    DummyVecEnv,
    make_vec_env,
)


class MockEnv(gym.Env):
    """Mock environment for testing."""

    def __init__(self, observation_space=None, action_space=None):
        if observation_space is None:
            observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32
            )
        if action_space is None:
            action_space = gym.spaces.Discrete(2)

        self.observation_space = observation_space
        self.action_space = action_space
        self._step_count = 0
        self._max_steps = 10

    def reset(self, seed=None, options=None):
        self._step_count = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        reward = 1.0
        terminated = self._step_count >= self._max_steps
        truncated = False
        info = {"step": self._step_count}
        return obs, reward, terminated, truncated, info

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        pass

    def seed(self, seed=None):
        return [seed]


class TestVecEnv:
    """Test the abstract VecEnv class."""

    def test_vec_env_is_abstract(self):
        """Test that VecEnv cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VecEnv(
                1, gym.spaces.Box(low=-1, high=1, shape=(4,)), gym.spaces.Discrete(2)
            )

    def test_seed_generation(self):
        """Test seed generation for multiple environments."""

        # Create a concrete implementation for testing
        class ConcreteVecEnv(VecEnv):
            def __init__(self, num_envs):
                super().__init__(
                    num_envs=num_envs,
                    observation_space=gym.spaces.Box(low=-1, high=1, shape=(4,)),
                    action_space=gym.spaces.Discrete(2),
                )

            def reset(self):
                pass

            def step_async(self, actions):
                pass

            def step_wait(self):
                pass

            def close(self):
                pass

            def get_attr(self, attr_name, indices=None):
                pass

            def set_attr(self, attr_name, value, indices=None):
                pass

            def env_method(self, method_name, *args, indices=None, **kwargs):
                # Mock the seed method call
                if method_name == "seed":
                    return [kwargs.get("seed", None)]
                return []

        vec_env = ConcreteVecEnv(3)

        # Test with seed
        seeds = vec_env.seed(42)
        assert len(seeds) == 3
        assert seeds == [42, 43, 44]

        # Test without seed
        seeds = vec_env.seed(None)
        assert len(seeds) == 3
        assert all(s is None for s in seeds)


class TestDummyVecEnv:
    """Test the DummyVecEnv implementation."""

    def test_initialization(self):
        """Test DummyVecEnv initialization."""
        env_fns = [lambda: MockEnv() for _ in range(3)]
        vec_env = DummyVecEnv(env_fns)

        assert vec_env.num_envs == 3
        assert isinstance(vec_env.observation_space, gym.spaces.Box)
        assert isinstance(vec_env.action_space, gym.spaces.Discrete)

    def test_duplicate_env_detection(self):
        """Test that duplicate environment instances are detected."""
        mock_env = MockEnv()
        env_fns = [lambda: mock_env for _ in range(2)]  # Same instance

        with pytest.raises(ValueError, match="cannot reuse the same environment"):
            DummyVecEnv(env_fns)

    def test_reset(self):
        """Test environment reset."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        obs = vec_env.reset()
        assert obs.shape == (2, 4)  # 2 envs, 4-dim observations
        assert obs.dtype == np.float32

    def test_step(self):
        """Test environment stepping."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        vec_env.reset()
        actions = np.array([0, 1])

        obs, rewards, dones, infos = vec_env.step(actions)

        assert obs.shape == (2, 4)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        assert all(isinstance(info, dict) for info in infos)

    def test_auto_reset(self):
        """Test automatic reset when episode ends."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        vec_env.reset()

        # Step enough times to trigger episode end
        for _ in range(12):  # More than max_steps in MockEnv
            actions = np.array([0, 1])
            obs, rewards, dones, infos = vec_env.step(actions)

        # Check that environments were auto-reset
        assert obs.shape == (2, 4)

    def test_async_step(self):
        """Test async stepping interface."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        vec_env.reset()
        actions = np.array([0, 1])

        vec_env.step_async(actions)
        obs, rewards, dones, infos = vec_env.step_wait()

        assert obs.shape == (2, 4)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2

    def test_get_set_attr(self):
        """Test getting and setting attributes."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        # Test getting attribute
        max_steps = vec_env.get_attr("_max_steps")
        assert len(max_steps) == 2
        assert all(step == 10 for step in max_steps)

        # Test setting attribute
        vec_env.set_attr("_max_steps", 20)
        max_steps = vec_env.get_attr("_max_steps")
        assert all(step == 20 for step in max_steps)

        # Test with indices
        vec_env.set_attr("_max_steps", 30, indices=[0])
        max_steps = vec_env.get_attr("_max_steps")
        assert max_steps[0] == 30
        assert max_steps[1] == 20

    def test_env_method(self):
        """Test calling methods on environments."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        # Test calling seed method
        results = vec_env.env_method("seed", seed=42)
        assert len(results) == 2

        # Test with indices
        results = vec_env.env_method("seed", seed=100, indices=[1])
        assert len(results) == 1

    def test_rendering(self):
        """Test environment rendering."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        # Test human mode (should not return anything)
        result = vec_env.render(mode="human")
        assert result is None

        # Test rgb_array mode
        images = vec_env.render(mode="rgb_array")
        assert images.shape == (2, 64, 64, 3)

    def test_get_images(self):
        """Test getting rendered images."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        images = vec_env.get_images()
        assert len(images) == 2
        assert all(img.shape == (64, 64, 3) for img in images)

    def test_unwrapped_property(self):
        """Test unwrapped property."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        unwrapped = vec_env.unwrapped
        assert len(unwrapped) == 2
        assert all(isinstance(env, MockEnv) for env in unwrapped)

    def test_close(self):
        """Test closing environments."""
        mock_envs = []

        def make_env():
            env = MockEnv()
            env.close = Mock()
            mock_envs.append(env)
            return env

        env_fns = [make_env for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        vec_env.close()

        # Check that close was called on all environments
        for env in mock_envs:
            env.close.assert_called_once()

    def test_dict_observation_space(self):
        """Test with dictionary observation space."""
        dict_obs_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),
                "velocity": gym.spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),
            }
        )

        class DictMockEnv(MockEnv):
            def __init__(self):
                super().__init__(observation_space=dict_obs_space)

            def reset(self, seed=None, options=None):
                self._step_count = 0
                obs = self.observation_space.sample()
                return obs, {}

            def step(self, action):
                self._step_count += 1
                obs = self.observation_space.sample()
                reward = 1.0
                terminated = self._step_count >= self._max_steps
                truncated = False
                info = {"step": self._step_count}
                return obs, reward, terminated, truncated, info

        env_fns = [lambda: DictMockEnv() for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        obs = vec_env.reset()
        assert isinstance(obs, dict)
        assert "position" in obs
        assert "velocity" in obs
        assert obs["position"].shape == (2, 2)
        assert obs["velocity"].shape == (2, 2)


class TestMakeVecEnv:
    """Test the make_vec_env utility function."""

    def test_make_vec_env_with_env_id(self):
        """Test make_vec_env with environment ID."""
        vec_env = make_vec_env("CartPole-v1", n_envs=2, seed=42)

        assert isinstance(vec_env, DummyVecEnv)
        assert vec_env.num_envs == 2

        vec_env.close()

    def test_make_vec_env_with_callable(self):
        """Test make_vec_env with callable."""
        vec_env = make_vec_env(lambda: MockEnv(), n_envs=2, seed=42)

        assert isinstance(vec_env, DummyVecEnv)
        assert vec_env.num_envs == 2

        vec_env.close()

    def test_make_vec_env_with_wrapper(self):
        """Test make_vec_env with wrapper class."""

        class TestWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.wrapped = True

        vec_env = make_vec_env(lambda: MockEnv(), n_envs=2, wrapper_class=TestWrapper)

        # Check that wrapper was applied
        wrapped_attrs = vec_env.get_attr("wrapped")
        assert all(attr for attr in wrapped_attrs)

        vec_env.close()

    def test_make_vec_env_with_custom_vec_env_cls(self):
        """Test make_vec_env with custom vectorized environment class."""
        vec_env = make_vec_env(lambda: MockEnv(), n_envs=2, vec_env_cls=DummyVecEnv)

        assert isinstance(vec_env, DummyVecEnv)
        vec_env.close()


class TestVecEnvWrapper:
    """Test the VecEnvWrapper base class."""

    def test_wrapper_delegation(self):
        """Test that wrapper delegates calls to wrapped environment."""
        env_fns = [lambda: MockEnv() for _ in range(2)]
        base_vec_env = DummyVecEnv(env_fns)

        # Create a simple wrapper
        class TestVecEnvWrapper(VecEnvWrapper):
            pass

        wrapped_vec_env = TestVecEnvWrapper(base_vec_env)

        # Test that properties are delegated
        assert wrapped_vec_env.num_envs == base_vec_env.num_envs
        assert wrapped_vec_env.observation_space == base_vec_env.observation_space
        assert wrapped_vec_env.action_space == base_vec_env.action_space

        # Test that methods are delegated
        obs = wrapped_vec_env.reset()
        assert obs.shape == (2, 4)

        actions = np.array([0, 1])
        obs, rewards, dones, infos = wrapped_vec_env.step(actions)
        assert obs.shape == (2, 4)

        wrapped_vec_env.close()


if __name__ == "__main__":
    pytest.main([__file__])
