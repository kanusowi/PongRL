import pygame
# import sys # sys.exit might not be needed if env.close() handles pygame.quit()
from game.pong_env import PongEnv # Use the environment

def human_play_loop():
    env = PongEnv(render_mode='human')
    observation, info = env.reset() # Initial reset

    running = True
    while running and env.screen is not None: # Continue if env screen is active
        action = 0 # Default: Stay still

        # Event handling should be done within PongEnv._render_frame() if it's active.
        # However, for direct key presses for actions, we can poll here.
        # The QUIT event handling in PongEnv should set env.screen to None.
        for event in pygame.event.get(): # Poll events to catch QUIT if not handled by env._render_frame
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
        if not running: break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = 1 # Up
        elif keys[pygame.K_s]:
            action = 2 # Down

        if not env.screen: # If env.close() was called due to internal event
            running = False
            break

        observation, reward, terminated, truncated, info = env.step(action)

        # PongEnv's step() in 'human' mode already calls _render_frame and clock.tick().

        if terminated: # A point was scored
             print(f"Point! Score: Player {info['player_score']} - Opponent {info['opponent_score']}")
             # The environment automatically resets the ball. Loop continues for the next point.

    env.close() # Ensure environment resources are released

if __name__ == "__main__":
    human_play_loop()
