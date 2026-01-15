import torch
import torch.nn as nn
import torch.nn.functional as F


class MapCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(MapCNN, self).__init__()
        # Entrada: (Canales, Alto, Ancho) -> Ej: (3, 10, 10)
        self.conv1 = nn.Conv2d(
            input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Calculamos el tamaño de salida de las convulsiones
        self.flatten_dim = 32 * input_shape[1] * input_shape[2]

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        # Salida: Probabilidad de acciones
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
