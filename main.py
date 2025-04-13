import pygame
import argparse

from game.pong_env import PongEnv
from utils.helpers import set_global_seeds
from config import POINTS_TO_WIN_GAME, SEED_PLAY

def human_vs_ai_game_loop(seed_value=SEED_PLAY, points_to_win=POINTS_TO_WIN_GAME, human_controls_opponent=False):
    set_global_seeds(seed_value)

    # Initialize the environment in 'human' render mode.
    # The 'human_opponent' flag in PongEnv determines if the right paddle listens to keyboard input
    # (K_UP, K_DOWN) or uses its internal simple AI.
    env = PongEnv(render_mode='human', human_opponent=human_controls_opponent)
    
    opponent_type = "Human" if human_controls_opponent else "Simple AI"
    print(f"\n--- Human vs. {opponent_type} Pong Game ---")
    print(f"  First to {points_to_win} points wins.")
    print("  Player 1 (Left): Use 'W' (Up) and 'S' (Down).")
    if human_controls_opponent:
        print("  Player 2 (Right): Use 'Up Arrow' (Up) and 'Down Arrow' (Down).")
    print("  Press ESCAPE to quit at any time.")
    print("-----------------------------------\n")

    current_observation, game_info = env.reset(seed=seed_value) # Initial reset of the environment
    
    game_running = True
    # Main game loop
    while game_running and env.screen is not None:
        
        player1_action = 0 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game_running = False
                    break
        if not game_running: break 

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player1_action = 1 
        elif keys[pygame.K_s]: 
            player1_action = 2 
            
        if not env.screen:
            game_running = False
            break
        current_observation, reward, terminated, truncated, info = env.step(player1_action)
                
        if terminated: 
            print(f"  Point! Score: Player {info['player_score']} - Opponent {info['opponent_score']}")
            if info['player_score'] >= points_to_win or info['opponent_score'] >= points_to_win:
                game_running = False
    
    if env.screen: 
        final_player_score = env.player_score
        final_opponent_score = env.opponent_score
        print(f"\nGame Finished. Final Score: Player {final_player_score} - Opponent {final_opponent_score}")
        if final_player_score > final_opponent_score:
            print("  Player 1 (Left Paddle) wins!")
        elif final_opponent_score > final_player_score:
            print("  Opponent (Right Paddle) wins!")
        else:
            print("  It's a draw (if points_to_win allows)!")
    env.close()
    print("Exiting Pong game.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Pong - Human vs. AI or Human vs. Human.")
    parser.add_argument("--seed", type=int, default=SEED_PLAY, help=f"Random seed for the game. Default: {SEED_PLAY}")
    parser.add_argument("--points", type=int, default=POINTS_TO_WIN_GAME, help=f"Points needed to win a game. Default: {POINTS_TO_WIN_GAME}")
    parser.add_argument("--human_opponent", action='store_true', help="If set, the right paddle is controlled by a human (Up/Down Arrows). Default is AI opponent.")
    args = parser.parse_args()
    human_vs_ai_game_loop(seed_value=args.seed, points_to_win=args.points, human_controls_opponent=args.human_opponent)
