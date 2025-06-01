# game/ball.py
import pygame
import random
from config import (WHITE, SCREEN_WIDTH, SCREEN_HEIGHT, BALL_RADIUS,BALL_BASE_SPEED_X, BALL_BASE_SPEED_Y, BALL_SPEED_INCREASE_FACTOR, BALL_MAX_SPEED_Y_FACTOR)

class Ball(pygame.sprite.Sprite):
    RADIUS = BALL_RADIUS

    def __init__(self, initial_x_center, initial_y_center, color=WHITE):
        super().__init__()

        self.image = pygame.Surface([BALL_RADIUS * 2, BALL_RADIUS * 2])
        self.image.set_colorkey(BLACK)
        self.image.fill(BLACK)
        pygame.draw.circle(self.image, color, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        
        self.rect = self.image.get_rect()
        self.rect.centerx = initial_x_center
        self.rect.centery = initial_y_center
        
        self.base_speed_x = BALL_BASE_SPEED_X
        self.base_speed_y = BALL_BASE_SPEED_Y
        
        self.current_speed_x_magnitude = self.base_speed_x
        
        # velocity main
        self.velocity_x = 0
        self.velocity_y = 0
        self.reset_ball() # initial velocity; random

    def update(self, player_paddle, opponent_paddle):
        self.rect.x += self.velocity_x
        self.rect.y += self.velocity_y

        # wall
        if self.rect.top <= 0:
            self.rect.top = 0 
            self.velocity_y *= -1
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT 
            self.velocity_y *= -1
        
        # collision
        paddles = [player_paddle, opponent_paddle]
        for paddle in paddles:
            if self.rect.colliderect(paddle.rect):
                is_player_paddle = (paddle == player_paddle)
                moving_towards_player_paddle = self.velocity_x < 0
                moving_towards_opponent_paddle = self.velocity_x > 0

                if (is_player_paddle and moving_towards_player_paddle) or \
                   (not is_player_paddle and moving_towards_opponent_paddle):
                    if is_player_paddle:
                        self.rect.left = paddle.rect.right
                    else:
                        self.rect.right = paddle.rect.left

                    self.current_speed_x_magnitude *= BALL_SPEED_INCREASE_FACTOR
                    self.velocity_x = self.current_speed_x_magnitude * (-1 if self.velocity_x > 0 else 1)

                    if not is_player_paddle: 
                        self.velocity_x = -abs(self.velocity_x) 
                    else:
                        self.velocity_x = abs(self.velocity_x)

                    offset_y = self.rect.centery - paddle.rect.centery
                    normalized_hit_pos = offset_y / (paddle.rect.height / 2)      
                    self.velocity_y = self.base_speed_y * normalized_hit_pos * 1.5
                    max_y_speed = self.base_speed_y * BALL_MAX_SPEED_Y_FACTOR
                    self.velocity_y = max(-max_y_speed, min(max_y_speed, self.velocity_y))

                    break

    def reset_ball(self, direction_to_serve=None):
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.centery = SCREEN_HEIGHT // 2
        self.current_speed_x_magnitude = self.base_speed_x # reset speed magnitude
        if direction_to_serve is None:
            direction_to_serve = random.choice((1, -1))        
        self.velocity_x = self.current_speed_x_magnitude * direction_to_serve
        self.velocity_y = self.base_speed_y * random.choice((1, -1)) * random.uniform(0.7, 1.3)

    def draw(self, screen):
        screen.blit(self.image, self.rect)