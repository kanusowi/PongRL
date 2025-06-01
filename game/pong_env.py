# game/pong_env.py
import pygame
import numpy as np
import random
from config import (SCREEN_WIDTH, SCREEN_HEIGHT, FPS, WHITE, BLACK, RED, GRAY, PADDLE_OFFSET, BALL_BASE_SPEED_X, BALL_BASE_SPEED_Y, UI_FONT_TYPE, UI_FONT_SIZE_SCORE)
from .paddle import Paddle
from .ball import Ball

class PongEnv:
    def __init__(self, render_mode=None, human_opponent=False, opponent_speed_factor=0.8):
        pygame.init()
        pygame.font.init()

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.fps = FPS
        self.screen = None
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Pong RL Environment")
        
        self.clock = pygame.time.Clock()
        self.player_paddle = Paddle(PADDLE_OFFSET, self.screen_height // 2, WHITE)
        self.opponent_paddle = Paddle(self.screen_width - PADDLE_OFFSET - Paddle.WIDTH, self.screen_height // 2, RED)
        self.ball = Ball(self.screen_width // 2, self.screen_height // 2)
        self.player_score = 0
        self.opponent_score = 0
        self.human_opponent = human_opponent
        self.opponent_ai_speed = int(self.opponent_paddle.speed * opponent_speed_factor)
        self.action_space_n = 3 

        dummy_obs, _ = self.reset()
        self.observation_space_shape = dummy_obs.shape


    def _get_obs(self):
        ball_x_norm = (self.ball.rect.centerx - self.screen_width / 2) / (self.screen_width / 2)
        ball_y_norm = (self.ball.rect.centery - self.screen_height / 2) / (self.screen_height / 2)
        ball_vx_norm = np.clip(self.ball.velocity_x / (BALL_BASE_SPEED_X * 2), -1, 1) 
        ball_vy_norm = np.clip(self.ball.velocity_y / (BALL_BASE_SPEED_Y * 2), -1, 1)
        
        player_paddle_y_norm = (self.player_paddle.rect.centery - self.screen_height / 2) / (self.screen_height / 2)
        opponent_paddle_y_norm = (self.opponent_paddle.rect.centery - self.screen_height / 2) / (self.screen_height / 2)
        
        obs = [
            ball_x_norm, ball_y_norm,
            ball_vx_norm, ball_vy_norm,
            player_paddle_y_norm,
            opponent_paddle_y_norm
        ]

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        return {"player_score": self.player_score, "opponent_score": self.opponent_score}

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.player_paddle.reset_position(self.screen_height // 2)
        self.opponent_paddle.reset_position(self.screen_height // 2)
        self.ball.reset_ball(direction_to_serve=random.choice((1, -1))) 
        self.player_score = 0
        self.opponent_score = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _handle_opponent_input(self):
        if not self.human_opponent: return
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.opponent_paddle.move(direction=-1)
        if keys[pygame.K_DOWN]:
            self.opponent_paddle.move(direction=1)

    def step(self, action):
        if action == 1:  # Up
            self.player_paddle.move(direction=-1)
        elif action == 2:  # Down
            self.player_paddle.move(direction=1)
        if self.human_opponent:
            self._handle_opponent_input()
        else: # ai opp
            dead_zone = self.opponent_paddle.rect.height * 0.1 # should help with jitter
            original_opponent_speed = self.opponent_paddle.speed
            self.opponent_paddle.speed = self.opponent_ai_speed # opp speed (AI)
            if self.opponent_paddle.rect.centery < self.ball.rect.centery - dead_zone:
                self.opponent_paddle.move(direction=1)
            elif self.opponent_paddle.rect.centery > self.ball.rect.centery + dead_zone:
                self.opponent_paddle.move(direction=-1)
            self.opponent_paddle.speed = original_opponent_speed
        self.ball.update(self.player_paddle, self.opponent_paddle)

        reward = 0.0
        terminated = False

        if self.ball.rect.left <= 0:
            self.opponent_score += 1
            reward = -1.0
            self.ball.reset_ball(direction_to_serve=1)
            terminated = True 

            self.player_score += 1
            reward = 1.0
            self.ball.reset_ball(direction_to_serve=-1)
            terminated = True
            
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._handle_pygame_events()
            if self.screen:
                self._render_frame()
                self.clock.tick(self.fps)

        return observation, reward, terminated, False, info

    def _handle_pygame_events(self):
        if not self.render_mode == 'human' or not self.screen:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def _render_frame(self):
        if self.screen is None: return
        self.screen.fill(BLACK)
        line_width = 4
        dash_height = 15
        gap_height = 10
        current_y = 0
        while current_y < self.screen_height:
            pygame.draw.rect(self.screen, GRAY, (self.screen_width // 2 - line_width // 2, current_y, line_width, dash_height))
            current_y += dash_height + gap_height
        
        # DRAW objects
        self.player_paddle.draw(self.screen)
        self.opponent_paddle.draw(self.screen)
        self.ball.draw(self.screen)
        
        # DRAW scores
        self._draw_text(f"{self.player_score}", UI_FONT_SIZE_SCORE, self.screen_width // 4, 20, WHITE)
        self._draw_text(f"{self.opponent_score}", UI_FONT_SIZE_SCORE, self.screen_width * 3 // 4, 20, WHITE)

        pygame.display.flip() 

    def _draw_text(self, text, size, x, y, color=WHITE, font_type=UI_FONT_TYPE):
        if not pygame.font.get_init(): pygame.font.init()
        try:
            font = pygame.font.Font(font_type, size)
        except FileNotFoundError:
            font = pygame.font.Font(pygame.font.match_font(font_type.lower()), size) 
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)        
        if self.screen:
            self.screen.blit(text_surface, text_rect)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None 
        if pygame.get_init(): 
            if pygame.font.get_init():
                pygame.font.quit()
            pygame.quit()