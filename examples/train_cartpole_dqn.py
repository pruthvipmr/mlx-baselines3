#!/usr/bin/env python3
"""
Train a DQN agent on CartPole-v1 environment.

This example demonstrates:
- Basic DQN training setup with experience replay
- Epsilon-greedy exploration with decay
- Target network updates for stable learning
- Model saving and loading
- Evaluation and visualization of results
"""

import argparse
import os
import sys
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from mlx_baselines3 import DQN
from mlx_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)


def make_env():
    """Create and return a CartPole-v1 environment."""
    return gym.make("CartPole-v1")


def train_dqn(
    total_timesteps: int = 50000,
    learning_rate: float = 1e-4,
    buffer_size: int = 50000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    tau: float = 1.0,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    max_grad_norm: float = 10.0,
    seed: int = 42,
    save_path: str = "dqn_cartpole",
    log_dir: str = "./logs/dqn_cartpole/",
    verbose: int = 1,
):
    """
    Train a DQN agent on CartPole-v1.
    
    Args:
        total_timesteps: Total number of timesteps to train for
        learning_rate: Learning rate for the optimizer
        buffer_size: Size of the replay buffer
        learning_starts: How many steps before learning starts
        batch_size: Minibatch size for each gradient update
        tau: Soft update coefficient for target network
        gamma: Discount factor
        train_freq: Update model every `train_freq` steps
        gradient_steps: Number of gradient steps after each rollout
        target_update_interval: Update target network every `target_update_interval` steps
        exploration_fraction: Fraction of timesteps for exploration
        exploration_initial_eps: Initial epsilon for exploration
        exploration_final_eps: Final epsilon for exploration
        max_grad_norm: Maximum norm for gradient clipping
        seed: Random seed
        save_path: Path to save the trained model
        log_dir: Directory for logging
        verbose: Verbosity level
    
    Returns:
        Trained DQN model
    """
    # Create environment
    env = make_env()
    
    # Create evaluation environment
    eval_env = make_env()
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
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
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="dqn_checkpoint",
    )
    
    # Stop training when the model reaches the reward threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=195.0,  # CartPole is considered solved at 195
        verbose=1,
    )
    
    callbacks = [eval_callback, checkpoint_callback, stop_callback]
    
    # Train the agent
    print(f"Training DQN on CartPole-v1 for {total_timesteps} timesteps...")
    print(f"Exploration: {exploration_initial_eps} → {exploration_final_eps} over {exploration_fraction * total_timesteps:.0f} steps")
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
    model = DQN.load(model_path)
    
    # Create environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    # Evaluate the model
    episode_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use deterministic=True for evaluation (no exploration)
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


def analyze_exploration(model_path: str, n_steps: int = 1000):
    """
    Analyze the exploration behavior of a trained model.
    
    Args:
        model_path: Path to the saved model
        n_steps: Number of steps to analyze
    """
    # Load the trained model
    model = DQN.load(model_path)
    
    # Create environment
    env = make_env()
    
    # Analyze exploration vs exploitation
    obs, _ = env.reset()
    exploration_actions = 0
    total_actions = 0
    
    for step in range(n_steps):
        # Get action with exploration
        action_explore, _ = model.predict(obs, deterministic=False)
        # Get action without exploration  
        action_exploit, _ = model.predict(obs, deterministic=True)
        
        if action_explore != action_exploit:
            exploration_actions += 1
        total_actions += 1
        
        obs, _, terminated, truncated, _ = env.step(action_explore)
        if terminated or truncated:
            obs, _ = env.reset()
    
    exploration_rate = exploration_actions / total_actions
    print(f"\nExploration Analysis over {n_steps} steps:")
    print(f"Exploration rate: {exploration_rate:.2%}")
    print(f"Current epsilon: {model.exploration_rate:.4f}")
    
    env.close()


def main():
    """Main function to run training and evaluation."""
    parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
    parser.add_argument(
        "--timesteps", type=int, default=50000,
        help="Total timesteps for training (default: 50000)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=50000,
        help="Replay buffer size (default: 50000)"
    )
    parser.add_argument(
        "--exploration-fraction", type=float, default=0.1,
        help="Fraction of timesteps for exploration (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-path", type=str, default="dqn_cartpole",
        help="Path to save the model (default: dqn_cartpole)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs/dqn_cartpole/",
        help="Directory for logs (default: ./logs/dqn_cartpole/)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate existing model, don't train"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Number of episodes for evaluation (default: 100)"
    )
    parser.add_argument(
        "--analyze-exploration", action="store_true",
        help="Analyze exploration behavior of trained model"
    )
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Only evaluate existing model
        if not os.path.exists(f"{args.save_path}.zip"):
            print(f"Model {args.save_path}.zip not found. Train a model first.")
            return
        
        print(f"Evaluating model: {args.save_path}")
        evaluate_model(args.save_path, args.eval_episodes, args.render)
        
        if args.analyze_exploration:
            analyze_exploration(args.save_path)
    else:
        # Train the model
        model = train_dqn(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            exploration_fraction=args.exploration_fraction,
            seed=args.seed,
            save_path=args.save_path,
            log_dir=args.log_dir,
        )
        
        # Evaluate the trained model
        print("\nEvaluating trained model...")
        evaluate_model(args.save_path, args.eval_episodes)
        
        if args.analyze_exploration:
            analyze_exploration(args.save_path)


if __name__ == "__main__":
    main()
