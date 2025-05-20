import pygame # For event handling if PongEnv doesn't fully capture QUIT during play
import torch # For device selection
import numpy as np # Not strictly needed here if agent handles all np conversions
import time
import argparse

from game.pong_env import PongEnv # The game environment
from agent.dqn_agent import DQNAgent # The agent that will play

def watch_agent(model_path="models/dqn_pong_final.pth",
                num_games=3, points_per_game=5, render_delay=0.02, seed=42):
    """Loads a trained DQN agent and lets it play Pong for visualization.

    Params
    ======
        model_path (str): Path to the saved model weights file.
        num_games (int): Number of full games to play.
        points_per_game (int): Number of points required to win a game.
        render_delay (float): Delay in seconds between frames to slow down rendering.
        seed (int): Random seed for environment (e.g., for consistent ball serves if desired for demo).
    """
    env = PongEnv(render_mode='human') # Ensure rendering is enabled
    # Initialize environment with a seed for consistent play if needed
    initial_obs_tuple = env.reset(seed=seed)
    state_size = len(initial_obs_tuple[0]) # Get observation size from env
    action_size = env.action_space_n       # Get action size from env

    # Initialize agent (hyperparameters like network size must match the trained model)
    # The seed for the agent during play doesn't usually matter as much if epsilon is 0.
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed)

    try:
        agent.load_model(model_path) # Use agent's load_model method
        print(f"Successfully loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Agent will use initial (random) weights.")
    except Exception as e: # Catch other potential errors during model loading
        print(f"Error loading model: {e}. Agent will use initial (random) weights.")

    agent.epsilon = 0.0 # Set epsilon to 0 for deterministic (greedy) actions during play

    for i_game in range(1, num_games + 1):
        print(f"\n--- Starting Game {i_game} of {num_games} ---")
        # Reset environment for a new game, possibly with a new seed for variety
        current_state, game_info = env.reset(seed=seed + i_game)

        # Scores for the current game
        game_player_score = 0
        game_opponent_score = 0

        game_running = True
        while game_running and env.screen is not None: # Continue if env window is active
            # Agent chooses action based on current state
            action = agent.act(current_state) # Epsilon is 0, so it's greedy

            # Environment takes a step
            next_state, reward, terminated, _, info = env.step(action)

            current_state = next_state # Update state

            # PongEnv's step() in 'human' mode handles rendering and its own clock.tick().
            # An additional delay can be added here if the game runs too fast to observe.
            time.sleep(render_delay)

            if terminated: # A point was scored
                game_player_score = info["player_score"]
                game_opponent_score = info["opponent_score"]
                print(f"Point! Current Game Score: Player {game_player_score} - Opponent {game_opponent_score}")

                # Check if the game score limit has been reached
                if game_player_score >= points_per_game or game_opponent_score >= points_per_game:
                    game_running = False # End this game
                # If not, the ball has already been reset by PongEnv, and the loop continues for the next point.

            # Check if user closed the Pygame window (PongEnv.close() sets env.screen to None)
            if not env.screen:
                print("Game window closed by user. Exiting current game.")
                game_running = False # Stop current game
                break # Break from inner point loop

        if not env.screen: # If window closed, exit outer game loop as well
             print("Exiting play session.")
             break

        print(f"Game {i_game} finished. Final Score: Player {game_player_score} - Opponent {game_opponent_score}")
        if game_player_score > game_opponent_score: print("Player wins!")
        elif game_opponent_score > game_player_score: print("Opponent wins!")
        else: print("It's a draw (if points per game allows)!")

    print("\nFinished watching agent.")
    env.close() # Clean up environment resources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Pong.")
    parser.add_argument("--model", type=str, default="models/dqn_pong_final.pth",
                        help="Path to the trained model file (e.g., models/dqn_pong_final.pth).")
    parser.add_argument("--games", type=int, default=3,
                        help="Number of full games to watch.")
    parser.add_argument("--points", type=int, default=5,
                        help="Points needed to win a game.")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Delay in seconds between frames to slow down rendering.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the environment during play (for consistency if needed).")
    args = parser.parse_args()

    watch_agent(model_path=args.model,
                num_games=args.games,
                points_per_game=args.points,
                render_delay=args.delay,
                seed=args.seed)
