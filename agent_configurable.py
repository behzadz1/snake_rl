"""
Configurable DQN Agent for Snake RL Experiments
Supports variable memory buffer sizes and neural network architectures
"""
import torch
import random
import numpy as np
from collections import deque
from game_headless import SnakeGameHeadless, Direction, Point
from model_configurable import ConfigurableQNet, QTrainer, ARCHITECTURES

# Default hyperparameters
DEFAULT_BATCH_SIZE = 1000
DEFAULT_LR = 0.001
DEFAULT_GAMMA = 0.9


class ConfigurableAgent:
    """
    DQN Agent with configurable architecture and memory.
    
    Args:
        architecture: Either a string key from ARCHITECTURES or a list of layer sizes
        memory_size: Size of experience replay buffer
        batch_size: Training batch size
        lr: Learning rate
        gamma: Discount factor
    """
    
    def __init__(self, architecture='baseline', memory_size=100000, 
                 batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR, gamma=DEFAULT_GAMMA):
        
        self.n_games = 0
        self.epsilon = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        # Setup memory buffer
        self.memory = deque(maxlen=memory_size)
        
        # Setup neural network
        if isinstance(architecture, str):
            if architecture not in ARCHITECTURES:
                raise ValueError(f"Unknown architecture: {architecture}")
            layer_sizes = ARCHITECTURES[architecture]
        else:
            layer_sizes = architecture
            
        self.architecture_name = architecture if isinstance(architecture, str) else 'custom'
        self.model = ConfigurableQNet(layer_sizes)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)
        
        # Metrics tracking
        self.losses = []

    def get_state(self, game):
        """Extract 11-feature state from game."""
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train on batch from replay memory."""
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.losses.append(loss)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on single step (online learning)."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Select action using epsilon-greedy policy."""
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def get_config_string(self):
        """Return configuration summary string."""
        return (f"Architecture: {self.model.get_architecture_string()}, "
                f"Memory: {self.memory_size}, Batch: {self.batch_size}")
