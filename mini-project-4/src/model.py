import torch
import torch.nn as nn

class FashionClassifier(nn.Module):
    def __init__(self, hidden_sizes=[256, 128], dropout=0.3):
        super().__init__()

        layers = []
        input_size = 784

        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden

        layers.append(nn.Linear(input_size, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.network(x)
