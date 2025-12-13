"""
Configurable Neural Network for DQN Snake Agent
Supports variable layer configurations for architecture experiments
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class ConfigurableQNet(nn.Module):
    """
    Flexible Q-Network supporting variable architectures.
    
    Args:
        layer_sizes: List of layer sizes, e.g., [11, 256, 3] or [11, 256, 128, 64, 3]
                    First element is input size, last is output size.
    """
    
    def __init__(self, layer_sizes):
        super().__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output sizes")
        
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        
        # Create layers dynamically
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x):
        # Apply ReLU to all layers except the last
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        # Output layer (no activation for Q-values)
        x = self.layers[-1](x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def get_architecture_string(self):
        """Return string representation of architecture."""
        return "→".join(map(str, self.layer_sizes))


class QTrainer:
    """Q-Learning trainer with configurable learning rate and discount factor."""
    
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# Predefined architecture configurations for experiments
# Format: [input, hidden1, hidden2, ..., output]
# Hidden layer count = total layers - 2 (input and output)
ARCHITECTURES = {
    'baseline': [11, 256, 3],                                    # 1 hidden layer (standard)
    'wide': [11, 512, 3],                                        # 1 hidden layer (wider)
    'deep_4hidden': [11, 256, 256, 128, 64, 3],                  # 4 hidden layers
    'deep_6hidden': [11, 256, 256, 256, 128, 128, 64, 3],        # 6 hidden layers (5-7 range)
}
