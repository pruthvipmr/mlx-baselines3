"""
Performance tests for experience buffers.

This module tests that buffers meet throughput requirements as specified in the
technical specification.
"""

import time
import numpy as np
import gymnasium as gym

from mlx_baselines3.common.buffers import ReplayBuffer, RolloutBuffer


class TestBufferPerformance:
    """Test buffer performance to meet spec requirements."""

    def test_replay_buffer_throughput(self):
        """Test ReplayBuffer sampling meets ≥50k samples/s throughput."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        buffer_size = 100000
        n_envs = 4

        buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=obs_space,
            action_space=action_space,
            n_envs=n_envs,
        )

        # Fill buffer with random data
        print(f"Filling buffer with {buffer_size * n_envs} transitions...")
        for i in range(buffer_size):
            obs = np.random.randn(n_envs, 64).astype(np.float32)
            next_obs = np.random.randn(n_envs, 64).astype(np.float32)
            action = np.random.randn(n_envs, 8).astype(np.float32)
            reward = np.random.randn(n_envs).astype(np.float32)
            done = np.random.rand(n_envs) < 0.1  # 10% episode termination
            infos = [{} for _ in range(n_envs)]

            buffer.add(obs, next_obs, action, reward, done, infos)

        # Warm up
        for _ in range(10):
            buffer.sample(1024)

        # Measure sampling throughput
        batch_size = 1024
        num_samples = 100
        total_samples = batch_size * num_samples

        print(f"Measuring throughput for {total_samples} samples...")
        start_time = time.time()

        for _ in range(num_samples):
            batch = buffer.sample(batch_size)
            # Ensure the sampling actually completes (touch the data)
            _ = batch["observations"].shape

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = total_samples / elapsed

        print(f"ReplayBuffer throughput: {throughput:.0f} samples/s")
        print("Target: ≥50,000 samples/s")

        # Check if we meet the requirement
        assert throughput >= 50000, f"Throughput {throughput:.0f} < 50,000 samples/s"

    def test_rollout_buffer_batch_efficiency(self):
        """Test RolloutBuffer batch generation efficiency."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
        action_space = gym.spaces.Discrete(4)
        buffer_size = 2048
        n_envs = 8

        buffer = RolloutBuffer(
            buffer_size=buffer_size,
            observation_space=obs_space,
            action_space=action_space,
            n_envs=n_envs,
        )

        # Fill buffer
        for i in range(buffer_size):
            obs = np.random.randn(n_envs, 32).astype(np.float32)
            action = np.random.randint(0, 4, size=n_envs)
            reward = np.random.randn(n_envs).astype(np.float32)
            episode_start = np.zeros(n_envs, dtype=np.bool_)
            value = np.random.randn(n_envs).astype(np.float32)
            log_prob = np.random.randn(n_envs).astype(np.float32)

            buffer.add(obs, action, reward, episode_start, value, log_prob)

        # Compute advantages
        last_values = np.random.randn(n_envs).astype(np.float32)
        dones = np.zeros(n_envs, dtype=np.bool_)
        buffer.compute_returns_and_advantage(last_values, dones)

        # Measure batch generation time
        batch_size = 512
        num_batches = 50
        total_samples = 0

        start_time = time.time()

        for batch in buffer.get(batch_size):
            total_samples += batch["observations"].shape[0]
            if total_samples >= num_batches * batch_size:
                break

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = total_samples / elapsed

        print(f"RolloutBuffer batch throughput: {throughput:.0f} samples/s")

        # Rollout buffer should also be reasonably fast (less strict requirement)
        assert throughput >= 10000, f"Rollout throughput {throughput:.0f} too slow"

    def test_memory_optimized_replay_buffer(self):
        """Test memory optimized ReplayBuffer performance."""
        obs_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        action_space = gym.spaces.Discrete(4)
        buffer_size = 10000

        # Test both regular and memory optimized versions
        for optimize_memory in [False, True]:
            buffer = ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=obs_space,
                action_space=action_space,
                optimize_memory_usage=optimize_memory,
            )

            # Fill buffer partially
            num_transitions = min(1000, buffer_size)
            for i in range(num_transitions):
                obs = np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8)
                next_obs = np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8)
                action = np.array([np.random.randint(0, 4)])
                reward = np.array([np.random.randn()])
                done = np.array([np.random.rand() < 0.1])
                infos = [{}]

                buffer.add(obs, next_obs, action, reward, done, infos)

            # Measure sampling time
            batch_size = 32
            num_samples = 100

            start_time = time.time()
            for _ in range(num_samples):
                batch = buffer.sample(batch_size)
                _ = batch["observations"].shape  # Touch the data

            end_time = time.time()
            elapsed = end_time - start_time
            throughput = (batch_size * num_samples) / elapsed

            print(f"Memory optimized={optimize_memory}: {throughput:.0f} samples/s")

            # Should still be reasonably fast even with large observations
            assert throughput >= 1000, f"Large obs throughput {throughput:.0f} too slow"

    def test_vectorized_env_efficiency(self):
        """Test buffer efficiency with different numbers of vectorized environments."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer_size = 5000

        # Test with different numbers of environments
        for n_envs in [1, 4, 16]:
            buffer = ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=obs_space,
                action_space=action_space,
                n_envs=n_envs,
            )

            # Fill buffer
            num_steps = buffer_size // 2
            for i in range(num_steps):
                obs = np.random.randn(n_envs, 16).astype(np.float32)
                next_obs = np.random.randn(n_envs, 16).astype(np.float32)
                action = np.random.randint(0, 2, size=n_envs)
                reward = np.random.randn(n_envs).astype(np.float32)
                done = np.random.rand(n_envs) < 0.05
                infos = [{} for _ in range(n_envs)]

                buffer.add(obs, next_obs, action, reward, done, infos)

            # Measure sampling performance
            batch_size = min(256, buffer.size())
            num_samples = 50

            start_time = time.time()
            for _ in range(num_samples):
                batch = buffer.sample(batch_size)
                _ = batch["observations"].shape

            end_time = time.time()
            elapsed = end_time - start_time
            throughput = (batch_size * num_samples) / elapsed

            print(f"n_envs={n_envs}: {throughput:.0f} samples/s")

            # Performance should scale reasonably with vectorization
            assert throughput >= 5000, (
                f"Vectorized performance {throughput:.0f} too slow"
            )


if __name__ == "__main__":
    # Run performance tests manually for detailed output
    test = TestBufferPerformance()
    print("Testing ReplayBuffer throughput...")
    test.test_replay_buffer_throughput()
    print("\nTesting RolloutBuffer efficiency...")
    test.test_rollout_buffer_batch_efficiency()
    print("\nTesting memory optimization...")
    test.test_memory_optimized_replay_buffer()
    print("\nTesting vectorized environment support...")
    test.test_vectorized_env_efficiency()
    print("\nAll performance tests completed!")
