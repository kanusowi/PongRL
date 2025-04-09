import pygame
from config import WHITE, SCREEN_HEIGHT, PADDLE_SPEED, PADDLE_HEIGHT, PADDLE_WIDTH # Use from config

class Paddle(pygame.sprite.Sprite):
    PADDLE_WIDTH_CLASS_ATTR = PADDLE_WIDTH  # Make accessible as class attribute
    PADDLE_HEIGHT_CLASS_ATTR = PADDLE_HEIGHT # Make accessible as class attribute

    def __init__(self, x, y_center, color=WHITE):
        super().__init__()
        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT]) # Use module/config const
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.centery = y_center
        self.speed = PADDLE_SPEED # Use module/config const

    def move(self, direction):
        """Moves the paddle up or down.
        direction: 1 for down, -1 for up.
        """
        if direction == 1: # Move down
            self.rect.y += self.speed
        elif direction == -1: # Move up
            self.rect.y -= self.speed

        # Keep paddle on screen (boundary check)
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT: # Use config const
            self.rect.bottom = SCREEN_HEIGHT

    def draw(self, screen):
        screen.blit(self.image, self.rect)
