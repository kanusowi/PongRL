import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, WHITE, BLACK, RED
from game.paddle import Paddle

def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pong RL")
    clock = pygame.time.Clock()

    player_paddle = Paddle(30, SCREEN_HEIGHT // 2, WHITE)
    # Use the class attribute for width if available, otherwise config or hardcoded
    opponent_paddle = Paddle(SCREEN_WIDTH - 30 - Paddle.PADDLE_WIDTH_CLASS_ATTR, SCREEN_HEIGHT // 2, RED)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Player paddle movement based on W/S keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player_paddle.move(direction=-1) # Move up
        if keys[pygame.K_s]:
            player_paddle.move(direction=1)  # Move down

        # Game logic updates (opponent AI, ball movement etc. later)

        # Drawing
        screen.fill(BLACK)
        player_paddle.draw(screen)
        opponent_paddle.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    game_loop()
