import pygame
import torch 
import time
import argparse
import os

from game.pong_env import PongEnv
from agent.dqn_agent import DQNAgent
from utils.helpers import set_global_seeds
from config import (NUM_GAMES_PLAY, POINTS_PER_GAME_PLAY, RENDER_DELAY_PLAY, 
                    SEED_PLAY, DEFAULT_MODEL_NAME, MODEL_SAVE_DIR)

def watch_trained_agent(model_filename=DEFAULT_MODEL_NAME,
                        num_games=NUM_GAMES_PLAY,
                        points_to_win=POINTS_PER_GAME_PLAY,
                        frame_delay=RENDER_DELAY_PLAY,
                        seed_value=SEED_PLAY):
    """
    Loads a trained DQN agent and runs it in the Pong environment for observation.

    Args:
        model_filename (str): Filename of the trained model to load from the `MODEL_SAVE_DIR`.
        num_games (int): Number of full games to play.
        points_to_win (int): Number of points required to win a single game.
        frame_delay (float): Delay in seconds between frames to make gameplay observable.
        seed_value (int): Random seed for environment reproducibility.
    """
    set_global_seeds(seed_value) # Ensure reproducibility for the game session

    # Initialize the environment in 'human' render mode to display the game
    env = PongEnv(render_mode='human')
    
    # Get state and action space dimensions
    initial_state, _ = env.reset(seed=seed_value)
    state_size = env.observation_space_shape[0]
    action_size = env.action_space_n
    
    # Initialize a new DQNAgent instance. Its weights will be overwritten by the loaded model.
    # The seed for the agent here primarily affects any internal random processes if they existed
    # beyond model weights (e.g., if exploration was still active, which it isn't in play mode).
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed_value)
    
    # Attempt to load the trained model weights
    full_model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    try:
        agent.load_model(model_filename)
        print(f"Successfully loaded trained model from: {full_model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {full_model_path}.")
        print("Agent will use initial (random) weights. Please train the agent first or check the model path.")
        env.close()
        return
    except Exception as e:
        print(f"Error loading model from {full_model_path}: {e}")
        print("Agent will use initial (random) weights.")
        # env.close() # Allow continuing with random agent for observation if desired, or exit.
                      # Current behavior: continues with an untrained agent if loading fails but file exists.
        
    # Set epsilon to 0.0 for deterministic (greedy) action selection based on learned policy
    agent.epsilon = 0.0 # Ensures the agent always picks the action with the highest Q-value

    print(f"\n--- Watching Trained Agent Play Pong ---")
    print(f"  Model: {model_filename}")
    print(f"  Playing {num_games} game(s), first to {points_to_win} points wins a game.")
    print(f"  Frame Delay: {frame_delay}s")
    print(f"  Seed: {seed_value}")
    print(f"  Using Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print("---------------------------------------\n")

    # Loop for the specified number of games
    for i_game in range(1, num_games + 1):
        print(f"--- Starting Game {i_game} of {num_games} ---")
        # Reset environment for each new game, using a different seed for variety if desired
        current_state, game_info = env.reset(seed=seed_value + i_game)
        game_player_score = 0 # Tracks score for the current full game
        game_opponent_score = 0
        
        game_running = True
        # Loop for a single game (until points_to_win is reached or window closed)
        while game_running and env.screen is not None: # env.screen check handles window close
            # Agent selects action based on the current state using its learned policy (epsilon=0)
            action = agent.act(current_state, eps=0.0)
            
            # Environment takes a step based on the agent's action
            next_state, reward, terminated, _, info = env.step(action)
            
            current_state = next_state # Update current state
            
            time.sleep(frame_delay) # Pause for human observation

            if terminated: # A point was scored
                game_player_score = info["player_score"]
                game_opponent_score = info["opponent_score"]
                print(f"  Point! Score: Player {game_player_score} - Opponent {game_opponent_score}")
                
                # Check if the game (not just a point/episode) is over
                if game_player_score >= points_to_win or game_opponent_score >= points_to_win:
                    game_running = False # End the current game
            
            # Handle manual window close during a point
            if not env.screen: 
                print("  Game window closed by user. Ending current game.")
                game_running = False # This will break the inner while loop
                break # Exit point loop immediately
        
        # If window was closed, break the outer game loop as well
        if not env.screen:
             print("Exiting play session due to window closure.")
             break 

        # Print results of the completed game
        print(f"Game {i_game} finished. Final Score: Player {game_player_score} - Opponent {game_opponent_score}")
        if game_player_score > game_opponent_score: print("  Player (Agent) wins the game!")
        elif game_opponent_score > game_player_score: print("  Opponent wins the game!")
        else: print("  The game is a draw (if points_to_win allows this, though typically Pong doesn't end in draws)!")
        print("---------------------------------------")


    print("\nFinished watching all games.")
    env.close() # Clean up Pygame resources

if __name__ == "__main__":
    # --- Argument Parsing for play.py ---
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Pong.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Filename of the trained model (in '{MODEL_SAVE_DIR}/' dir). Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--games", type=int, default=NUM_GAMES_PLAY, 
                        help=f"Number of full games to watch. Default: {NUM_GAMES_PLAY}")
    parser.add_argument("--points", type=int, default=POINTS_PER_GAME_PLAY, 
                        help=f"Points needed to win a game. Default: {POINTS_PER_GAME_PLAY}")
    parser.add_argument("--delay", type=float, default=RENDER_DELAY_PLAY, 
                        help=f"Delay in seconds between frames for slower rendering. Default: {RENDER_DELAY_PLAY}")
    parser.add_argument("--seed", type=int, default=SEED_PLAY, 
                        help=f"Random seed for the environment during play. Default: {SEED_PLAY}")
    args = parser.parse_args()

    watch_trained_agent(model_filename=args.model, 
                        num_games=args.games, 
                        points_to_win=args.points, 
                        frame_delay=args.delay,
                        seed_value=args.seed)
