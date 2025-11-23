import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden, num_classes):
        super().__init__()
        layers = []
        cur = input_dim
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
