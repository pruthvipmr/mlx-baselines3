#!/usr/bin/env python3
"""
Train a SAC agent on Pendulum-v1 environment.

This example demonstrates:
- SAC training for continuous control
- Automatic entropy temperature tuning
- Twin critic networks for reduced overestimation bias
- Model saving and loading
- Environment normalization for better performance
- Advanced evaluation metrics
"""

import argparse
import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from mlx_baselines3 import SAC
from mlx_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mlx_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)


def make_env():
    """Create and return a Pendulum-v1 environment."""
    return gym.make("Pendulum-v1")


def train_sac(
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 64,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    ent_coef: str = "auto",
    target_entropy: str = "auto",
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    use_normalization: bool = True,
    seed: int = 42,
    save_path: str = "sac_pendulum",
    log_dir: str = "./logs/sac_pendulum/",
    verbose: int = 1,
):
    """
    Train a SAC agent on Pendulum-v1.

    Args:
        total_timesteps: Total number of timesteps to train for
        learning_rate: Learning rate for actor and critic networks
        buffer_size: Size of the replay buffer
        learning_starts: How many steps before learning starts
        batch_size: Minibatch size for each gradient update
        tau: Soft update coefficient for target networks
        gamma: Discount factor
        train_freq: Update model every `train_freq` steps
        gradient_steps: Number of gradient steps after each rollout
        ent_coef: Entropy regularization coefficient ("auto" for automatic tuning)
        target_entropy: Target entropy for automatic entropy tuning
        use_sde: Whether to use State Dependent Exploration
        sde_sample_freq: Sample a new noise matrix every n steps
        use_normalization: Whether to use VecNormalize wrapper
        seed: Random seed
        save_path: Path to save the trained model
        log_dir: Directory for logging
        verbose: Verbosity level

    Returns:
        Trained SAC model and environment
    """
    # Create environment
    env = DummyVecEnv([make_env])

    # Apply normalization if requested
    if use_normalization:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
        )

    # Create evaluation environment (without normalization for fair comparison)
    eval_env = make_env()

    # Create model
    model = SAC(
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
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
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
        name_prefix="sac_checkpoint",
    )

    # Pendulum goal is typically around -200 to -150 (higher is better)
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=-150.0,
        verbose=1,
    )

    callbacks = [eval_callback, checkpoint_callback, stop_callback]

    # Train the agent
    print(f"Training SAC on Pendulum-v1 for {total_timesteps} timesteps...")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Using normalization: {use_normalization}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
    )

    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Save environment if using normalization
    if use_normalization:
        env.save(f"{save_path}_vecnormalize.pkl")
        print(f"VecNormalize stats saved to {save_path}_vecnormalize.pkl")

    return model, env


def evaluate_model(
    model_path: str,
    n_episodes: int = 100,
    render: bool = False,
    use_normalization: bool = True,
):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        use_normalization: Whether to load VecNormalize stats

    Returns:
        Mean reward and standard deviation
    """
    # Load the trained model
    model = SAC.load(model_path)

    # Create environment
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)

    # Load normalization if used during training
    if use_normalization:
        try:
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(f"{model_path}_vecnormalize.pkl", vec_env)
            vec_env.training = False  # Don't update stats during evaluation
            vec_env.norm_reward = False  # Don't normalize rewards during evaluation
            use_vec_env = True
        except FileNotFoundError:
            print("VecNormalize file not found, evaluating without normalization")
            use_vec_env = False
    else:
        use_vec_env = False

    # Evaluate the model
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        if use_vec_env:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            if use_vec_env:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = vec_env.step(action)
                reward = reward[0]  # Extract from array
                done = done[0]
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render and not use_vec_env:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{n_episodes}, Reward: {episode_reward:.2f}, "
                f"Length: {episode_length}"
            )

    if not use_vec_env:
        env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.1f}")

    return mean_reward, std_reward


def analyze_policy(model_path: str, n_samples: int = 1000):
    """
    Analyze the learned policy behavior.

    Args:
        model_path: Path to the saved model
        n_samples: Number of samples for analysis
    """
    # Load the trained model
    model = SAC.load(model_path)

    # Create environment
    env = make_env()

    # Collect observations and actions
    observations = []
    actions = []
    deterministic_actions = []

    obs, _ = env.reset()

    for _ in range(n_samples):
        observations.append(obs.copy())

        # Get stochastic action
        action_stoch, _ = model.predict(obs, deterministic=False)
        actions.append(action_stoch)

        # Get deterministic action
        action_det, _ = model.predict(obs, deterministic=True)
        deterministic_actions.append(action_det)

        obs, _, terminated, truncated, _ = env.step(action_stoch)
        if terminated or truncated:
            obs, _ = env.reset()

    observations = np.array(observations)
    actions = np.array(actions)
    deterministic_actions = np.array(deterministic_actions)

    print(f"\nPolicy Analysis over {n_samples} samples:")
    print(f"Observation range: [{observations.min():.2f}, {observations.max():.2f}]")
    print(f"Action range (stochastic): [{actions.min():.2f}, {actions.max():.2f}]")
    print(
        "Action range (deterministic): "
        f"[{deterministic_actions.min():.2f}, {deterministic_actions.max():.2f}]"
    )
    print(f"Action std (stochastic): {actions.std():.3f}")
    print(f"Action std (deterministic): {deterministic_actions.std():.3f}")

    # Check policy stochasticity
    action_diff = np.abs(actions - deterministic_actions).mean()
    print(
        "Mean difference between stochastic and deterministic actions: "
        f"{action_diff:.3f}"
    )

    env.close()


def main():
    """Main function to run training and evaluation."""
    parser = argparse.ArgumentParser(description="Train SAC on Pendulum-v1")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps for training (default: 100000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size (default: 100000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--ent-coef",
        type=str,
        default="auto",
        help="Entropy coefficient (default: auto)",
    )
    parser.add_argument(
        "--no-normalization", action="store_true", help="Disable VecNormalize wrapper"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="sac_pendulum",
        help="Path to save the model (default: sac_pendulum)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/sac_pendulum/",
        help="Directory for logs (default: ./logs/sac_pendulum/)",
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
    parser.add_argument(
        "--analyze-policy",
        action="store_true",
        help="Analyze policy behavior of trained model",
    )

    args = parser.parse_args()

    use_normalization = not args.no_normalization

    if args.eval_only:
        # Only evaluate existing model
        if not os.path.exists(f"{args.save_path}.zip"):
            print(f"Model {args.save_path}.zip not found. Train a model first.")
            return

        print(f"Evaluating model: {args.save_path}")
        evaluate_model(
            args.save_path, args.eval_episodes, args.render, use_normalization
        )

        if args.analyze_policy:
            analyze_policy(args.save_path)
    else:
        # Train the model
        model, env = train_sac(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            use_normalization=use_normalization,
            seed=args.seed,
            save_path=args.save_path,
            log_dir=args.log_dir,
        )

        # Evaluate the trained model
        print("\nEvaluating trained model...")
        evaluate_model(
            args.save_path, args.eval_episodes, use_normalization=use_normalization
        )

        if args.analyze_policy:
            analyze_policy(args.save_path)


if __name__ == "__main__":
    main()
