import pygame
from config import WHITE, SCREEN_HEIGHT # Assuming these are in config

# Default paddle dimensions and speed (can be moved to config.py later)
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 100
PADDLE_SPEED = 7 # This speed is per frame

class Paddle(pygame.sprite.Sprite): # Inherit from Sprite for potential group management
    def __init__(self, x, y_center, color=WHITE):
        super().__init__() # Initialize the sprite
        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.centery = y_center # Position paddle by its center y
        self.speed = PADDLE_SPEED

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    # Movement methods will be added in a subsequent commit
