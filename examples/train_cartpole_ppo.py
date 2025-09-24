#!/usr/bin/env python3
"""
Train a PPO agent on CartPole-v1 environment.

This example demonstrates:
- Basic PPO training setup
- Model saving and loading
- Evaluation and visualization of results
- Callbacks for monitoring training progress
"""

import argparse
import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from mlx_baselines3 import PPO
from mlx_baselines3.common.vec_env import DummyVecEnv
from mlx_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)


def make_env():
    """Create and return a CartPole-v1 environment."""
    return gym.make("CartPole-v1")


def train_ppo(
    total_timesteps: int = 10000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    save_path: str = "ppo_cartpole",
    log_dir: str = "./logs/ppo_cartpole/",
    verbose: int = 1,
):
    """
    Train a PPO agent on CartPole-v1.

    Args:
        total_timesteps: Total number of timesteps to train for
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs for policy update
        gamma: Discount factor
        gae_lambda: Factor for trade-off between bias and variance
        clip_range: Clipping parameter for surrogate loss
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: Maximum norm for gradient clipping
        seed: Random seed
        save_path: Path to save the trained model
        log_dir: Directory for logging
        verbose: Verbosity level

    Returns:
        Trained PPO model
    """
    # Create environment
    env = DummyVecEnv([make_env])

    # Create evaluation environment
    eval_env = make_env()

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        verbose=verbose,
    )

    # Create callbacks
    os.makedirs(log_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=2000,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=log_dir,
        name_prefix="ppo_checkpoint",
    )

    # Stop training when the model reaches the reward threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=195.0,  # CartPole is considered solved at 195
        verbose=1,
    )

    callbacks = [eval_callback, checkpoint_callback, stop_callback]

    # Train the agent
    print(f"Training PPO on CartPole-v1 for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
    )

    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model


def evaluate_model(model_path: str, n_episodes: int = 100, render: bool = False):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment

    Returns:
        Mean reward and standard deviation
    """
    # Load the trained model
    model = PPO.load(model_path)

    # Create environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    # Evaluate the model
    episode_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}, Reward: {episode_reward}")

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    return mean_reward, std_reward


def main():
    """Main function to run training and evaluation."""
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="Total timesteps for training (default: 20000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="ppo_cartpole",
        help="Path to save the model (default: ppo_cartpole)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/ppo_cartpole/",
        help="Directory for logs (default: ./logs/ppo_cartpole/)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing model, don't train",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment during evaluation"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation (default: 100)",
    )

    args = parser.parse_args()

    if args.eval_only:
        # Only evaluate existing model
        if not os.path.exists(f"{args.save_path}.zip"):
            print(f"Model {args.save_path}.zip not found. Train a model first.")
            return

        print(f"Evaluating model: {args.save_path}")
        evaluate_model(args.save_path, args.eval_episodes, args.render)
    else:
        # Train the model
        train_ppo(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            save_path=args.save_path,
            log_dir=args.log_dir,
        )

        # Evaluate the trained model
        print("\nEvaluating trained model...")
        evaluate_model(args.save_path, args.eval_episodes)


if __name__ == "__main__":
    main()
