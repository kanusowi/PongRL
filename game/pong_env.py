import pygame
import numpy as np
import random
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, WHITE, BLACK, RED, BALL_SPEED_X, BALL_SPEED_Y # Import speeds for obs normalization
from .paddle import Paddle   # Relative import
from .ball import Ball     # Relative import

class PongEnv:
    def __init__(self, render_mode=None):
        pygame.init()
        pygame.font.init() # Initialize font module if drawing text

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.fps = FPS

        self.screen = None
        if render_mode == 'human':
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Pong Env RL")

        self.clock = pygame.time.Clock()

        # Game objects
        self.player_paddle = Paddle(30, self.screen_height // 2, WHITE)
        self.opponent_paddle = Paddle(self.screen_width - 30 - Paddle.PADDLE_WIDTH_CLASS_ATTR, self.screen_height // 2, RED)
        self.ball = Ball(self.screen_width // 2, self.screen_height // 2) # Ball.py must exist

        self.player_score = 0
        self.opponent_score = 0
        self.render_mode = render_mode

        # For RL agent: action space (0: Stay, 1: Up, 2: Down for player paddle)
        self.action_space_n = 3

    def _get_obs(self):
        # Normalize observations to be roughly between -1 and 1 or 0 and 1
        obs = [
            (self.ball.rect.centerx - self.screen_width / 2) / (self.screen_width / 2),
            (self.ball.rect.centery - self.screen_height / 2) / (self.screen_height / 2),
            self.ball.velocity_x / BALL_SPEED_X, # Assumes BALL_SPEED_X is max typical speed
            self.ball.velocity_y / BALL_SPEED_Y, # Assumes BALL_SPEED_Y is max typical speed
            (self.player_paddle.rect.centery - self.screen_height / 2) / (self.screen_height / 2),
            (self.opponent_paddle.rect.centery - self.screen_height / 2) / (self.screen_height / 2)
        ]
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        return {"player_score": self.player_score, "opponent_score": self.opponent_score}

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed) # For consistent ball serves etc.
            # np.random.seed(seed) # If using numpy randomness for other parts

        # Reset paddle positions
        self.player_paddle.rect.centery = self.screen_height // 2
        self.opponent_paddle.rect.centery = self.screen_height // 2
        # Reset ball (Ball class needs reset_ball method)
        self.ball.reset_ball(direction=random.choice((1, -1)))

        self.player_score = 0
        self.opponent_score = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame() # Render initial state

        return observation, info # Gym-like return

    def step(self, action):
        # Player action
        if action == 1: # Up
            self.player_paddle.move(direction=-1)
        elif action == 2: # Down
            self.player_paddle.move(direction=1)
        # action == 0 is "stay still"

        # Opponent AI (simple: follows ball's y position with a deadzone)
        dead_zone = 5 # To prevent jittering
        if self.opponent_paddle.rect.centery < self.ball.rect.centery - dead_zone:
            self.opponent_paddle.move(direction=1) # move down
        elif self.opponent_paddle.rect.centery > self.ball.rect.centery + dead_zone:
            self.opponent_paddle.move(direction=-1) # move up

        # Ball update (Ball class needs update method taking paddles for collision)
        self.ball.update(self.player_paddle, self.opponent_paddle)

        reward = 0.0 # Use float for rewards
        terminated = False # True if an episode ends (e.g., point scored)

        # Scoring logic
        if self.ball.rect.left <= 0: # Opponent scores
            self.opponent_score += 1
            reward = -1.0
            self.ball.reset_ball(direction=1) # Serve to player
            terminated = True
        if self.ball.rect.right >= self.screen_width: # Player scores
            self.player_score += 1
            reward = 1.0
            self.ball.reset_ball(direction=-1) # Serve to opponent
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            self.clock.tick(self.fps) # Control game speed when rendering for humans

        # Gym-like return: obs, reward, terminated, truncated, info
        return observation, reward, terminated, False, info # False for truncated (not used here)

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human": # Initialize screen if not done
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Pong Env RL")

        if self.screen is None: return # Do nothing if no screen (e.g. training mode)

        self.screen.fill(BLACK)
        # Draw center line
        for i in range(0, self.screen_height, 25): # Dashed line
            pygame.draw.rect(self.screen, WHITE, (self.screen_width // 2 - 2, i, 4, 15)) # Thicker line

        self.player_paddle.draw(self.screen)
        self.opponent_paddle.draw(self.screen)
        self.ball.draw(self.screen)

        # Draw scores
        self._draw_text(f"Player: {self.player_score}", 36, self.screen_width // 4, 10)
        self._draw_text(f"Opponent: {self.opponent_score}", 36, self.screen_width * 3 // 4, 10)

        pygame.display.flip()

        # Handle Pygame events when rendering (e.g., to close window)
        # This is important to prevent the window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close() # Signal that the window was closed

    def _draw_text(self, text, size, x, y, color=WHITE):
        if not pygame.font.get_init(): pygame.font.init() # Ensure font module is ready
        # Consider using a specific font file for consistency: pygame.font.Font("path/to/font.ttf", size)
        font = pygame.font.Font(pygame.font.match_font('arial'), size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y) # Position by mid-top
        if self.screen: # Check if screen exists before drawing
            self.screen.blit(text_surface, text_rect)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit() # Important to quit font module
            pygame.quit()      # Quit all pygame modules
            self.screen = None # Mark screen as closed
