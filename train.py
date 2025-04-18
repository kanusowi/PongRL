import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import argparse
import torch

from game.pong_env import PongEnv
from agent.dqn_agent import DQNAgent
from utils.helpers import set_global_seeds
from config import (N_EPISODES_TRAIN, MAX_T_PER_POINT, PRINT_EVERY_TRAIN, DEFAULT_MODEL_NAME, TRAINING_PLOT_NAME, SOLVED_SCORE_THRESHOLD, SEED_TRAIN)

def train_dqn_agent(n_episodes=N_EPISODES_TRAIN, 
                    max_t_per_episode=MAX_T_PER_POINT, 
                    print_every=PRINT_EVERY_TRAIN, 
                    model_filename=DEFAULT_MODEL_NAME, 
                    plot_filename=TRAINING_PLOT_NAME,
                    solved_threshold=SOLVED_SCORE_THRESHOLD,
                    seed_value=SEED_TRAIN,
                    render_training=False):

    set_global_seeds(seed_value)

    render_mode_train = 'human' if render_training else None
    env = PongEnv(render_mode=render_mode_train)
    
    initial_state, _ = env.reset(seed=seed_value)
    state_size = env.observation_space_shape[0]
    action_size = env.action_space_n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed_value)

    scores_history = []
    scores_window = deque(maxlen=print_every)
    losses_window = deque(maxlen=print_every)
    
    print(f"\n--- Starting DQN Training for Pong ---")
    print(f"  Target Training Episodes (Points): {n_episodes}") 
    print(f"  Training Seed: {seed_value}")
    print(f"  State Space Size: {state_size}, Action Space Size: {action_size}")
    print(f"  Using PyTorch Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}") 
    print(f"  Models will be saved to: models/{model_filename} (and checkpoints)")
    print(f"  Training plot will be saved to: {plot_filename}")
    print("---------------------------------------------------\n") 

    for i_episode in range(1, n_episodes + 1):
        current_state, _ = env.reset(seed=seed_value + i_episode) 
        episode_score = 0 

        for t in range(max_t_per_episode):
            action = agent.act(current_state) # epsilon-greedy policy
            next_state, reward, terminated, _, info = env.step(action)            
            loss_value = agent.step(current_state, action, reward, next_state, terminated)
            
            if loss_value is not None:
                losses_window.append(loss_value)

            current_state = next_state
            episode_score += reward
            
            if terminated:
                break
        
        scores_window.append(episode_score)
        scores_history.append(episode_score)
        agent.update_epsilon()

        avg_score_over_window = np.mean(scores_window) if scores_window else 0.0
        avg_loss_over_window = np.mean(losses_window) if losses_window else 0.0
        
        print(f'\rEpisode {i_episode}/{n_episodes}\tAvg Score: {avg_score_over_window:.3f}\tAvg Loss: {avg_loss_over_window:.4f}\tEpsilon: {agent.epsilon:.4f}\tLast Score: {episode_score}   ', end="")
        
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}/{n_episodes}\tAvg Score: {avg_score_over_window:.3f}\tAvg Loss: {avg_loss_over_window:.4f}\tEpsilon: {agent.epsilon:.4f}\tLast Score: {episode_score}      ')
            checkpoint_name = model_filename.replace(".pth", f"_ckpt_{i_episode}.pth") # checkpoint model
            agent.save_model(checkpoint_name)

        # solved ?
        if len(scores_window) == print_every and avg_score_over_window >= solved_threshold:
            print(f'\n\nEnvironment solved in {i_episode} episodes!')
            print(f'Average Score over last {print_every} episodes: {avg_score_over_window:.3f}')
            agent.save_model(model_filename)
            break
            
    if not (len(scores_window) == print_every and avg_score_over_window >= solved_threshold):
        agent.save_model(model_filename)
        print(f"\n\nTraining finished after {n_episodes} episodes.")
        print(f"Final model saved to models/{model_filename}")

    env.close()
    return scores_history

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a DQN agent to play Pong.")
    parser.add_argument("--episodes", type=int, default=N_EPISODES_TRAIN,help=f"Number of training episodes (points scored). Default: {N_EPISODES_TRAIN}")
    parser.add_argument("--max_t", type=int, default=MAX_T_PER_POINT, help=f"Max timesteps per point. Default: {MAX_T_PER_POINT}")
    parser.add_argument("--print_every", type=int, default=PRINT_EVERY_TRAIN, help=f"Frequency to print average scores and save checkpoints. Default: {PRINT_EVERY_TRAIN}")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help=f"Filename for the saved model (in 'models/' dir). Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--plot_name", type=str, default=TRAINING_PLOT_NAME, help=f"Filename for the training scores plot. Default: {TRAINING_PLOT_NAME}")
    parser.add_argument("--solved_score", type=float, default=SOLVED_SCORE_THRESHOLD, help=f"Average score over 'print_every' episodes to consider solved. Default: {SOLVED_SCORE_THRESHOLD}")
    parser.add_argument("--seed", type=int, default=SEED_TRAIN,help=f"Random seed for training. Default: {SEED_TRAIN}")
    parser.add_argument("--render", action='store_true', help="Render the environment during training (slows down training significantly).") # Added help text detail
    args = parser.parse_args()

    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created 'models' directory for saved networks.")
        
    final_scores_history = train_dqn_agent(n_episodes=args.episodes, 
                                     max_t_per_episode=args.max_t,
                                     print_every=args.print_every,
                                     model_filename=args.model_name,
                                     plot_filename=args.plot_name,
                                     solved_threshold=args.solved_score,
                                     seed_value=args.seed,
                                     render_training=args.render) 

    if final_scores_history:
        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(1, len(final_scores_history) + 1), final_scores_history, label='Score per Episode', alpha=0.7)
        
        if len(final_scores_history) >= args.print_every:
            rolling_avg_values = [np.mean(final_scores_history[i-args.print_every:i]) for i in range(args.print_every, len(final_scores_history)+1)]
            plt.plot(np.arange(args.print_every, len(final_scores_history) + 1), rolling_avg_values, 
                     color='red', linewidth=2, label=f'Avg Score (Rolling {args.print_every} Episodes)')
        
        plt.ylabel('Score (Reward per Point/Episode)')
        plt.xlabel('Episode #')
        plt.title('DQN Agent Training Progress on Pong')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(args.plot_name)
        print(f"\nPlot of training scores saved to {args.plot_name}")
    else:
        print("No scores to plot (training might have been interrupted or very short).")
