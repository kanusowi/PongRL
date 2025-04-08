import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, WHITE, BLACK, RED
from game.paddle import Paddle # Import Paddle class

def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pong RL")
    clock = pygame.time.Clock()

    # Create paddles
    # Position paddles slightly off the edges, centered vertically
    # Using Paddle.PADDLE_WIDTH assumes it's a class attribute or defined in module scope
    player_paddle = Paddle(30, SCREEN_HEIGHT // 2, WHITE)
    opponent_paddle = Paddle(SCREEN_WIDTH - 30 - 15, SCREEN_HEIGHT // 2, RED) # Hardcode width for now if not class attr

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Game logic updates would go here

        # Drawing
        screen.fill(BLACK)
        player_paddle.draw(screen)
        opponent_paddle.draw(screen)
        # Other game elements drawing would go here

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    game_loop()
