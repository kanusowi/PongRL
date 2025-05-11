import pygame # Not strictly needed here unless for direct env event handling
import numpy as np
from collections import deque
import torch # For checking cuda availability or other torch utils
import matplotlib.pyplot as plt
import os # For creating models directory
import argparse # For command-line arguments

from game.pong_env import PongEnv # The game environment
from agent.dqn_agent import DQNAgent # The learning agent
# Configuration parameters (LR, GAMMA, etc.) are typically imported by DQNAgent itself from config.py
# or passed to its constructor.

def train_dqn(n_episodes=2000, max_t_per_point=500, print_every=100,
              model_save_path="models/dqn_pong_final.pth",
              plot_save_path="training_scores.png",
              solved_score_threshold=0.5, # Avg score over `print_every` episodes to consider "solved"
              seed=0): # Add seed for reproducibility
    """Trains a DQN agent on the Pong environment.

    Params
    ======
        n_episodes (int): maximum number of training episodes (points scored)
        max_t_per_point (int): maximum number of timesteps per point before forced reset (if no score)
        print_every (int): frequency of printing average scores and saving checkpoints
        model_save_path (str): path to save the final trained model
        plot_save_path (str): path to save the plot of training scores
        solved_score_threshold (float): average score over `print_every` episodes to consider solved
        seed (int): random seed for agent and environment
    """
    env = PongEnv(render_mode=None) # No human rendering during training for speed
    # Initialize environment with seed if its reset method supports it
    # For PongEnv, seed in reset affects ball serve direction
    initial_obs_tuple = env.reset(seed=seed)
    state_size = len(initial_obs_tuple[0]) # Observation is the first element of the tuple
    action_size = env.action_space_n

    # Agent will use hyperparameters from config.py by default, or can be passed
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed)

    scores_history = []                        # List containing scores from each episode (point)
    scores_window = deque(maxlen=print_every)  # Last 'print_every' scores for averaging

    print(f"Starting training for {n_episodes} episodes with seed {seed}...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")


    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + i_episode) # Vary seed per episode for diverse starts
        current_episode_score = 0 # Score for this specific point/episode

        # Loop for one point (one "episode" in this RL context)
        for t in range(max_t_per_point):
            action = agent.act(state) # Agent handles its own epsilon for exploration
            next_state, reward, terminated, _, info = env.step(action)

            # Agent's step method saves experience and learns if conditions met
            agent.step(state, action, reward, next_state, terminated)

            state = next_state
            current_episode_score += reward
            if terminated: # Point scored (episode ends)
                break

        scores_window.append(current_episode_score)
        scores_history.append(current_episode_score)
        agent.update_epsilon() # Decay epsilon after each episode (point)

        avg_score_over_window = np.mean(scores_window) if scores_window else 0.0
        # Clear previous line with \r and extra spaces for overwriting
        print(f'\rEpisode {i_episode}\tAvg Score (Last {print_every}): {avg_score_over_window:.2f}\tEpsilon: {agent.epsilon:.4f}\tLast Score: {current_episode_score}   ', end="")

        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tAvg Score (Last {print_every}): {avg_score_over_window:.2f}\tEpsilon: {agent.epsilon:.4f}\tLast Score: {current_episode_score}      ')
            if model_save_path: # Save checkpoint model
                 checkpoint_path = model_save_path.replace(".pth", f"_ckpt_{i_episode}.pth")
                 agent.save_model(checkpoint_path) # Use agent's save_model method

        # Check if the environment is considered "solved"
        if avg_score_over_window >= solved_score_threshold and len(scores_window) == print_every:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score_over_window:.2f}')
            if model_save_path: agent.save_model(model_save_path) # Save final "solved" model
            break # Stop training

    # Save final model if training completed all episodes (and not saved by "solved" condition)
    if not (avg_score_over_window >= solved_score_threshold and len(scores_window) == print_every):
        if model_save_path:
            agent.save_model(model_save_path)
            print(f"\nTraining finished after {n_episodes} episodes. Model saved to {model_save_path}")

    env.close() # Clean up environment resources
    return scores_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Pong.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes (points).")
    parser.add_argument("--max_t", type=int, default=500, help="Max timesteps per point before forced reset.")
    parser.add_argument("--print_every", type=int, default=100, help="Frequency to print average scores and save checkpoints.")
    parser.add_argument("--save_path", type=str, default="models/dqn_pong_final.pth", help="Path to save the final model.")
    parser.add_argument("--plot_path", type=str, default="training_scores.png", help="Path to save the training scores plot.")
    parser.add_argument("--solved_score", type=float, default=0.5, help="Average score over 'print_every' episodes to consider solved.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training.")
    args = parser.parse_args()

    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    final_scores = train_dqn(n_episodes=args.episodes,
                             max_t_per_point=args.max_t,
                             print_every=args.print_every,
                             model_save_path=args.save_path,
                             plot_save_path=args.plot_path,
                             solved_score_threshold=args.solved_score,
                             seed=args.seed)

    # Plotting the scores history
    if final_scores: # Check if training returned scores
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(final_scores)), final_scores, label='Score per Episode')
        # Calculate and plot rolling average for smoother trend
        if len(final_scores) >= args.print_every:
            rolling_avg = np.convolve(final_scores, np.ones(args.print_every)/args.print_every, mode='valid')
            plt.plot(np.arange(args.print_every -1, len(final_scores)), rolling_avg, color='red', label=f'Avg Score (Last {args.print_every})')

        plt.ylabel('Score (Reward per Point)')
        plt.xlabel('Episode #')
        plt.title('DQN Agent Training Progress on Pong')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.plot_path)
        print(f"Plot of training scores saved to {args.plot_path}")
        # plt.show() # Optionally display the plot interactively
