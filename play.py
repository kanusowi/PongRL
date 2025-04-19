import pygame
import torch 
import time
import argparse
import os

from game.pong_env import PongEnv
from agent.dqn_agent import DQNAgent
from utils.helpers import set_global_seeds
from config import (NUM_GAMES_PLAY, POINTS_PER_GAME_PLAY, RENDER_DELAY_PLAY, SEED_PLAY, DEFAULT_MODEL_NAME, MODEL_SAVE_DIR)

def watch_trained_agent(model_filename=DEFAULT_MODEL_NAME,
                        num_games=NUM_GAMES_PLAY,
                        points_to_win=POINTS_PER_GAME_PLAY,
                        frame_delay=RENDER_DELAY_PLAY,
                        seed_value=SEED_PLAY):
    
    set_global_seeds(seed_value) 

    env = PongEnv(render_mode='human')
    
    initial_state, _ = env.reset(seed=seed_value)
    state_size = env.observation_space_shape[0]
    action_size = env.action_space_n
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed_value)
    
    # load the trained model weights
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
    agent.epsilon = 0.0 

    print(f"\n--- Watching Trained Agent Play Pong ---")
    print(f"  Model: {model_filename}")
    print(f"  Playing {num_games} game(s), first to {points_to_win} points wins a game.")
    print(f"  Frame Delay: {frame_delay}s")
    print(f"  Seed: {seed_value}")
    print(f"  Using Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print("---------------------------------------\n")

    for i_game in range(1, num_games + 1):
        print(f"--- Starting Game {i_game} of {num_games} ---")
        current_state, game_info = env.reset(seed=seed_value + i_game)
        game_player_score = 0
        game_opponent_score = 0
        
        game_running = True
        while game_running and env.screen is not None:
            action = agent.act(current_state, eps=0.0)    
            next_state, reward, terminated, _, info = env.step(action)            
            current_state = next_state            
            time.sleep(frame_delay)

            if terminated:
                game_player_score = info["player_score"]
                game_opponent_score = info["opponent_score"]
                print(f"  Point! Score: Player {game_player_score} - Opponent {game_opponent_score}")
                
                if game_player_score >= points_to_win or game_opponent_score >= points_to_win:
                    game_running = False 
            
            if not env.screen: 
                print("  Game window closed by user. Ending current game.")
                game_running = False
                break 
        
        if not env.screen:
             print("Exiting play session due to window closure.")
             break 

        print(f"Game {i_game} finished. Final Score: Player {game_player_score} - Opponent {game_opponent_score}")
        if game_player_score > game_opponent_score: print("  Player (Agent) wins the game!")
        elif game_opponent_score > game_player_score: print("  Opponent wins the game!")
        else: print("  The game is a draw (if points_to_win allows this, though typically Pong doesn't end in draws)!")
        print("---------------------------------------")


    print("\nFinished watching all games.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Pong.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help=f"Filename of the trained model (in '{MODEL_SAVE_DIR}/' dir). Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--games", type=int, default=NUM_GAMES_PLAY, help=f"Number of full games to watch. Default: {NUM_GAMES_PLAY}")
    parser.add_argument("--points", type=int, default=POINTS_PER_GAME_PLAY, help=f"Points needed to win a game. Default: {POINTS_PER_GAME_PLAY}")
    parser.add_argument("--delay", type=float, default=RENDER_DELAY_PLAY, help=f"Delay in seconds between frames for slower rendering. Default: {RENDER_DELAY_PLAY}")
    parser.add_argument("--seed", type=int, default=SEED_PLAY, help=f"Random seed for the environment during play. Default: {SEED_PLAY}")
    args = parser.parse_args()

    watch_trained_agent(model_filename=args.model, 
                        num_games=args.games, 
                        points_to_win=args.points, 
                        frame_delay=args.delay,
                        seed_value=args.seed)
