import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_layers=None):
        super().__init__()

        if hidden_layers is None:
            self.out = nn.Linear(input_dim, num_classes)
            self.hidden = []
        else:
            self.hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
            for i in range(1, len(hidden_layers)):
                self.hidden.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i])])
            self.out = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x):

        for hd in self.hidden:
            x = F.relu(hd(x))

        x = F.log_softmax(self.out(x))
        return x
