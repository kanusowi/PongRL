# Screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)

# Paddle
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 100
PADDLE_SPEED = 8  # paddle speed (pixelperframe)
PADDLE_OFFSET = 30 # distance from edge of screen

# Ball
BALL_RADIUS = 8
BALL_BASE_SPEED_X = 6 
BALL_BASE_SPEED_Y = 6  
BALL_SPEED_INCREASE_FACTOR = 1.05 
BALL_MAX_SPEED_Y_FACTOR = 1.5 

# Game 
POINTS_TO_WIN_GAME = 5

# Replay Buffer
BUFFER_SIZE = int(1e5) 
BATCH_SIZE = 64        

# DQN Hyperparameters
GAMMA = 0.99 # discount factor
TAU = 1e-3   # QNet soft update - target
LR = 5e-4   # adam
UPDATE_EVERY = 4  # update frequency

# QNet Architecture (model.py)
FC1_UNITS = 128
FC2_UNITS = 128

# Epsilon Params
EPS_START = 1.0   
EPS_END = 0.01    
EPS_DECAY = 0.995 

# Training (train.py)
N_EPISODES_TRAIN = 3000       
MAX_T_PER_POINT = 700        
PRINT_EVERY_TRAIN = 100      
SOLVED_SCORE_THRESHOLD = 0.7 
SEED_TRAIN = 0               

# Play (play.py)
NUM_GAMES_PLAY = 3              
POINTS_PER_GAME_PLAY = 5        
RENDER_DELAY_PLAY = 0.02   # frame delay
SEED_PLAY = 42  # seed 

# Files
MODEL_SAVE_DIR = "models"
DEFAULT_MODEL_NAME = "dqn_pong_final.pth"
TRAINING_PLOT_NAME = "training_scores_pong.png"

# Misc
UI_FONT_TYPE = 'arial' 
UI_FONT_SIZE_SCORE = 48
UI_FONT_SIZE_INFO = 24