# PongRL: Learn Deep Q-Networks with Pygame & PyTorch

Welcome to PongRL! This project provides a clear and educational implementation of a Deep Q-Network (DQN) agent that learns to play the classic game of Pong. It's designed for beginners and enthusiasts looking to understand and experiment with fundamental concepts in Deep Reinforcement Learning.

The codebase is heavily commented to explain not just *what* the code does, but *why* certain design choices are made from an RL perspective.

## Core Goal: Learning to Play Pong
The primary objective is to train an AI agent that can:
1.  Observe the game state (ball position, paddle positions, ball velocity).
2.  Decide how to move its paddle (up, down, or stay).
3.  Learn through trial and error (rewards for scoring, penalties for being scored upon) to become a proficient Pong player.

## Key Features & Learning Opportunities

-   **Complete Pong Game Environment**: Built from scratch using **Pygame**. It offers a simple API (`reset`, `step`) similar to popular RL environments like Gymnasium, making it easy to integrate with RL agents.
    -   *Learn about:* Game loop mechanics, sprite management, collision detection (`game/pong_env.py`, `game/ball.py`, `game/paddle.py`).
-   **Deep Q-Network (DQN) Agent**: Implemented in **PyTorch**. This is where the "Deep Learning" meets "Reinforcement Learning."
    -   **Q-Network (`agent/model.py`)**: A neural network (Multi-Layer Perceptron) that learns to predict the expected future rewards (Q-values) for taking different actions in different states.
        -   *Learn about:* Neural network architecture for RL, input/output design, activation functions.
    -   **Experience Replay (`agent/replay_buffer.py`)**: A crucial DQN component. The agent stores its experiences (state, action, reward, next state, done flag) in a buffer and samples random mini-batches from it for training. This breaks temporal correlations and improves learning stability.
        -   *Learn about:* Off-policy learning, data storage for RL.
    -   **Target Network (`agent/dqn_agent.py`)**: A separate, periodically updated copy of the Q-Network used to provide stable targets during the Q-learning update. This helps prevent oscillations and divergence during training.
        -   *Learn about:* Stabilizing Q-learning, fixed Q-targets concept.
    -   **Epsilon-Greedy Exploration (`agent/dqn_agent.py`)**: The agent's strategy for balancing exploration (trying random actions to discover new strategies) and exploitation (choosing the best-known action). Epsilon (the probability of exploring) decays over time.
        -   *Learn about:* The exploration-exploitation dilemma.
-   **Model Persistence (`agent/dqn_agent.py`)**: Functionality to save trained model weights (`.pth` files) and load them for continued training or evaluation.
-   **Configurable Training Script (`train.py`)**:
    -   Manages the agent-environment interaction loop for learning.
    -   **Visual Training**: Includes a `--render` flag to watch the agent learn in real-time (though this significantly slows down training).
    -   **Detailed Logging**: Prints training progress (scores, average loss, epsilon) to the console.
    -   **Plotting**: Saves a plot of training scores using Matplotlib for visualizing learning progress.
    -   **Checkpoints**: Saves model checkpoints periodically.
    -   *Learn about:* Structuring an RL training loop, hyperparameter management via `config.py` and command-line arguments.
-   **Agent Evaluation Script (`play.py`)**:
    -   Loads a pre-trained agent model and renders it playing Pong.
    -   Allows you to see your trained agent in action!
-   **Human Play Modes (`main.py`)**:
    -   Play Pong yourself against the simple AI opponent.
    -   Play against another human player on the same keyboard.
-   **Heavily Commented Code**: Designed to be a learning resource.
-   **Centralized Configuration (`config.py`)**: All key game parameters and DQN hyperparameters are in one place, well-commented to encourage experimentation.

## Project Structure

```
PongRL/
├── .gitignore                 # Specifies intentionally untracked files
├── README.md                  # This file!
├── requirements.txt           # Python package dependencies
├── config.py                  # Global configurations: game settings, RL hyperparameters, file paths
├── main.py                    # Script for Human vs. AI or Human vs. Human gameplay
├── train.py                   # Script to train the DQN agent
├── play.py                    # Script to watch a trained agent play
├── game/                      # Modules for the Pong game environment
│   ├── __init__.py
│   ├── pong_env.py            # Core Pong game environment (Gym-like API, state, rewards)
│   ├── paddle.py              # Paddle class (movement, rendering)
│   └── ball.py                # Ball class (movement, collision physics)
├── agent/                     # Modules for the DQN Reinforcement Learning agent
│   ├── __init__.py
│   ├── dqn_agent.py           # DQNAgent class (learning algorithm, action selection, target updates)
│   ├── model.py               # PyTorch Q-Network model architecture (nn.Module)
│   └── replay_buffer.py       # Experience Replay Buffer implementation
├── utils/                     # Utility modules
│   ├── __init__.py
│   └── helpers.py             # Helper functions, e.g., `set_global_seeds`
├── models/                    # Directory for saved model weights (created automatically)
│   └── .gitkeep               # Ensures the directory is tracked
├── notes/                     # Directory for your development notes or research (optional)
│   └── .gitkeep
└── training_scores_pong.png   # Default name for the plot generated by train.py
```

## System Requirements
-   Python 3.7+
-   Pygame
-   NumPy
-   PyTorch (CPU or GPU version)
-   Matplotlib (for plotting training scores)

It is **highly recommended** to install these dependencies using the provided `requirements.txt` file, preferably within a Python virtual environment:

```bash
# 1. Create a virtual environment (optional, but good practice)
# python -m venv venv_pongrl  # Or any name you prefer
#
# 2. Activate the virtual environment:
#    On Linux/macOS:
#    source venv_pongrl/bin/activate
#    On Windows (Command Prompt/PowerShell):
#    .\venv_pongrl\Scripts\activate
#
# 3. Install dependencies:
pip install -r requirements.txt
```

## How to Use PongRL

### 1. Training Your DQN Agent
This is where the magic happens! The agent learns to play Pong from scratch.

To start training, run the `train.py` script from the root `PongRL/` directory:
```bash
python train.py
```
This will use default parameters from `config.py`.

**Customizing Training:**
You can override many parameters using command-line arguments. For example:
```bash
python train.py --episodes 5000 --model_name pong_expert_v1.pth --plot_name pong_expert_v1_scores.png --seed 123
```
-   `--episodes`: Number of training "episodes" (an episode ends when a point is scored).
-   `--model_name`: Filename for the final saved model (e.g., `my_model.pth`). Saved in the `models/` directory.
-   `--plot_name`: Filename for the training scores plot.
-   `--seed`: A random seed for reproducibility.
-   `--render`: Add this flag to watch the agent train (slows down training significantly!):
    ```bash
    python train.py --render
    ```

Run `python train.py --help` to see all available options and their default values.

**Interpreting Training Output:**
During training, you'll see output like:
`Episode 100/3000 Avg Score: -0.50 Avg Loss: 0.0123 Epsilon: 0.6065 Last Score: -1.0`
-   **Avg Score**: Average reward per point over the last `PRINT_EVERY_TRAIN` episodes. Positive means the agent is scoring more than the opponent.
-   **Avg Loss**: Average Q-Network loss. Should generally decrease or stabilize if learning is occurring.
-   **Epsilon**: Current exploration rate. Decreases over time.
-   **Last Score**: Reward from the most recent point (+1 for agent scoring, -1 for opponent scoring).

The training script will save model checkpoints periodically and a final model (e.g., `models/dqn_pong_final.pth`). A plot of scores (e.g., `training_scores_pong.png`) will also be saved.

### 2. Watching Your Trained Agent Play
Once you have a trained model (e.g., `models/dqn_pong_final.pth`), you can watch it play:
```bash
python play.py --model dqn_pong_final.pth
```
**Customizing Playback:**
```bash
python play.py --model your_model_name.pth --games 3 --points 5 --delay 0.05
```
-   `--model`: Name of the model file in the `models/` directory.
-   `--games`: Number of full games to play.
-   `--points`: Points needed to win a game.
-   `--delay`: Delay (in seconds) between frames for slower, more observable gameplay.

Run `python play.py --help` for all options.

### 3. Playing Pong Yourself (Human vs. AI / Human vs. Human)
Use `main.py` to play the game:

-   **Human vs. Simple AI (Default):**
    ```bash
    python main.py
    ```
    You control the left paddle with 'W' (up) and 'S' (down). The right paddle is a basic AI.

-   **Human vs. Human:**
    ```bash
    python main.py --human_opponent
    ```
    Player 1 (left): 'W'/'S'.
    Player 2 (right): 'Up Arrow'/'Down Arrow'.

Run `python main.py --help` for options like setting points to win or the game seed.

## Understanding the DQN Agent (A Deeper Dive)

This project implements a standard Deep Q-Network. Here's a brief overview of key concepts visible in the code:

-   **State (`game/pong_env.py -> _get_obs()`):** The agent "sees" the game as a vector of numbers: normalized ball position (x, y), ball velocity (vx, vy), and y-positions of both paddles. This is the input to its neural network.
-   **Actions (`game/pong_env.py -> action_space_n`):** The agent can choose one of three discrete actions: move its paddle up, move down, or stay still (no-op).
-   **Q-Network (`agent/model.py -> QNetwork`):** This neural network learns to estimate the "quality" (expected future discounted reward, or Q-value) of taking each possible action from a given state.
    -   Input: State vector.
    -   Output: Q-value for each action (e.g., Q(s, up), Q(s, down), Q(s, stay)).
-   **Epsilon-Greedy Policy (`agent/dqn_agent.py -> act()`):**
    -   With probability `epsilon`, the agent explores by choosing a random action.
    -   With probability `1-epsilon`, the agent exploits its current knowledge by choosing the action with the highest Q-value predicted by its Q-Network.
    -   `epsilon` starts high (e.g., 1.0) and decays over time, shifting from exploration to exploitation.
-   **Experience Replay (`agent/replay_buffer.py -> ReplayBuffer`):**
    -   The agent stores its experiences `(state, action, reward, next_state, done_flag)` in a large buffer.
    -   During learning, it samples random mini-batches of these experiences to train its Q-Network. This breaks correlations between consecutive experiences and reuses data efficiently.
-   **Learning Update (`agent/dqn_agent.py -> learn()`):**
    -   The core of DQN. It uses the Bellman equation to update Q-value estimates.
    -   **Target Q-value (Y):** `Y = reward + gamma * max_a' Q_target(next_state, a')` (if `next_state` is not terminal).
        -   `reward`: Immediate reward received.
        -   `gamma`: Discount factor (how much to value future rewards).
        -   `Q_target(next_state, a')`: Q-value of the best action `a'` in the `next_state`, estimated by the **target network**.
    -   **Loss:** The Mean Squared Error (MSE) between this target `Y` and the Q-value predicted by the **local Q-Network** for the action actually taken: `Loss = (Y - Q_local(state, action))^2`.
    -   The gradients of this loss are used to update the weights of the local Q-Network.
-   **Target Network (`agent/dqn_agent.py -> qnetwork_target`):**
    -   A copy of the local Q-Network whose weights are updated more slowly (e.g., "soft updates" or periodic hard copies).
    -   Using a separate, more stable target network for calculating `Q_targets_next` helps prevent oscillations and makes training more stable.

## Experimentation Ideas for Learners

-   **Hyperparameters (`config.py`):**
    -   Modify `LR` (learning rate), `GAMMA` (discount factor), `BATCH_SIZE`, `BUFFER_SIZE`.
    -   Adjust `EPS_DECAY`, `EPS_START`, `EPS_END` to change the exploration strategy.
    -   Change `UPDATE_EVERY` (how often the network learns).
    -   Modify `TAU` (target network soft update rate).
-   **Network Architecture (`agent/model.py` and `config.py`):**
    -   Change `FC1_UNITS`, `FC2_UNITS`. Add/remove layers.
    -   Try different activation functions (though ReLU is standard).
-   **State Representation (`game/pong_env.py -> _get_obs()`):**
    -   Can you add more information? Or remove some? How does it affect learning?
    -   What if you didn't normalize the values?
-   **Reward Shaping (Advanced - `game/pong_env.py -> step()`):**
    -   The current reward is sparse (+1 for scoring, -1 for being scored on).
    -   Could you add small intermediate rewards (e.g., for hitting the ball)? Be careful, as this can sometimes lead to unintended behaviors.
-   **Opponent AI (`game/pong_env.py`):**
    -   Can you make the simple AI opponent more challenging or behave differently?

## Contributing
This project is primarily for educational purposes. If you find bugs or have suggestions for improving clarity or educational value, feel free to open an issue or submit a pull request!

Happy Learning!
