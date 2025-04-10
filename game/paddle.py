import pygame
from config import WHITE, SCREEN_HEIGHT, PADDLE_SPEED, PADDLE_HEIGHT, PADDLE_WIDTH

class Paddle(pygame.sprite.Sprite):    
    WIDTH = PADDLE_WIDTH  
    HEIGHT = PADDLE_HEIGHT 

    def __init__(self, x, initial_y_center, color=WHITE):
        super().__init__() 

        self.image = pygame.Surface([self.WIDTH, self.HEIGHT])
        self.image.fill(color)        
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.centery = initial_y_center        
        self.speed = PADDLE_SPEED

    def move(self, direction):
        if direction == 1: 
            self.rect.y += self.speed
        elif direction == -1:
            self.rect.y -= self.speed
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

    def reset_position(self, y_center):
        self.rect.centery = y_center

    def draw(self, screen):
        screen.blit(self.image, self.rect)