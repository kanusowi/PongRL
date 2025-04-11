# ==============================================================================
# --- Game Display & Mechanics Configuration ---
# ==============================================================================
# Screen
SCREEN_WIDTH = 800  # Width of the game window in pixels
SCREEN_HEIGHT = 600 # Height of the game window in pixels
FPS = 60            # Frames per second for game rendering and logic updates

# Colors (RGB tuples)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)    # Typically used for the opponent paddle
GREEN = (0, 255, 0)  # General purpose color
BLUE = (0, 0, 255)   # General purpose color
GRAY = (100, 100, 100) # Used for the center dashed line

# Paddle
PADDLE_WIDTH = 15     # Width of the paddles
PADDLE_HEIGHT = 100   # Height of the paddles
PADDLE_SPEED = 8      # Speed of paddle movement in pixels per frame (for human/simple AI)
PADDLE_OFFSET = 30    # Distance of paddles from the side edges of the screen

# Ball
BALL_RADIUS = 8       # Radius of the ball
BALL_BASE_SPEED_X = 6 # Initial horizontal speed component of the ball
BALL_BASE_SPEED_Y = 6 # Initial vertical speed component of the ball
BALL_SPEED_INCREASE_FACTOR = 1.05 # Factor by which ball's X-speed increases on paddle hit
BALL_MAX_SPEED_Y_FACTOR = 1.5   # Max Y-speed relative to base Y-speed (limits extreme angles)

# Game Rules
POINTS_TO_WIN_GAME = 5 # Points needed for a player to win a full game (used in play.py and main.py)

# ==============================================================================
# --- DQN Agent Hyperparameters ---
# ==============================================================================
# These parameters significantly affect the agent's learning performance and stability.
# Finding optimal values often requires experimentation (hyperparameter tuning).

# Replay Buffer
BUFFER_SIZE = int(1e5) # Maximum number of experiences (state, action, reward, next_state, done) to store.
                       # Larger buffers can provide more diverse experiences but require more memory.
                       # Typical values: 1e4 to 1e6.
BATCH_SIZE = 64        # Number of experiences to sample from the buffer for each learning step.
                       # Larger batches provide more stable gradients but are computationally more expensive.
                       # Typical values: 32, 64, 128.

# DQN Core Hyperparameters
GAMMA = 0.99           # Discount factor for future rewards. Determines how much the agent values future rewards
                       # over immediate ones. Closer to 1 means more farsighted.
                       # Typical values: 0.9 to 0.999.
TAU = 1e-3             # Interpolation parameter for soft updating the target network.
                       # target_weights = TAU * local_weights + (1 - TAU) * target_weights.
                       # Smaller TAU means slower, more stable updates to the target network.
                       # Typical values: 1e-2 to 1e-4.
LR = 5e-4              # Learning rate for the Adam optimizer. Controls how much the network weights are
                       # adjusted during each optimization step.
                       # Typical values: 1e-3 to 1e-5.
UPDATE_EVERY = 4       # Frequency (in timesteps) at which the agent learns from the replay buffer.
                       # Learning every few steps can be more computationally efficient than every step.

# QNetwork Architecture (see agent/model.py)
FC1_UNITS = 128        # Number of units in the first fully connected hidden layer of the Q-network.
FC2_UNITS = 128        # Number of units in the second fully connected hidden layer.
                       # Network capacity should be sufficient for the complexity of the task.

# Epsilon-Greedy Exploration Parameters
EPS_START = 1.0        # Initial value of epsilon (probability of taking a random action).
                       # Starts with full exploration.
EPS_END = 0.01         # Minimum value of epsilon. Ensures some minimal exploration even late in training.
EPS_DECAY = 0.995      # Multiplicative factor by which epsilon is decayed after each episode.
                       # Controls how quickly the agent shifts from exploration to exploitation.

# ==============================================================================
# --- Training Configuration (for train.py) ---
# ==============================================================================
N_EPISODES_TRAIN = 3000       # Total number of episodes (points scored) to train the agent for.
MAX_T_PER_POINT = 700         # Maximum number of timesteps allowed per episode (point).
                              # Prevents episodes from running indefinitely if the ball gets stuck or agent doesn't learn.
PRINT_EVERY_TRAIN = 100       # Frequency (in episodes) to print training progress and save model checkpoints.
SOLVED_SCORE_THRESHOLD = 0.7  # Average score over `PRINT_EVERY_TRAIN` episodes to consider the environment "solved".
                              # For Pong, a positive score means the agent is winning more than losing.
SEED_TRAIN = 0                # Random seed for training to ensure reproducibility.

# ==============================================================================
# --- Gameplay Configuration (for play.py and main.py) ---
# ==============================================================================
NUM_GAMES_PLAY = 3            # Number of full games to play when watching a trained agent.
POINTS_PER_GAME_PLAY = 5      # Points needed to win a single game during evaluation in play.py.
RENDER_DELAY_PLAY = 0.02      # Delay (in seconds) between frames when rendering gameplay in play.py, for watchability.
SEED_PLAY = 42                # Random seed for playing/evaluation sessions.

# ==============================================================================
# --- File & Directory Names ---
# ==============================================================================
MODEL_SAVE_DIR = "models"                       # Directory to save trained model weights.
DEFAULT_MODEL_NAME = "dqn_pong_final.pth"       # Default filename for the final saved model.
TRAINING_PLOT_NAME = "training_scores_pong.png" # Default filename for the plot of training scores.

# ==============================================================================
# --- Miscellaneous UI & Other Settings ---
# ==============================================================================
UI_FONT_TYPE = 'arial'        # Font type for displaying scores and info text.
UI_FONT_SIZE_SCORE = 48       # Font size for the score display.
UI_FONT_SIZE_INFO = 24        # Font size for other informational text (not currently used extensively).
