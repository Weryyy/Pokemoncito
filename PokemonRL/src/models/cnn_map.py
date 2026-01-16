import torch
import torch.nn as nn
import torch.nn.functional as F

class MapCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(MapCNN, self).__init__()
        
        # input_shape será ahora (9, 10, 10) gracias al Frame Stacking
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculamos el tamaño de la capa lineal
        # 10x10 -> convs (padding=1 mantiene tamaño) -> 10x10
        self.fc_input_dim = 64 * input_shape[1] * input_shape[2]
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Aplanar
        x = F.relu(self.fc1(x))
        return self.fc2(x)