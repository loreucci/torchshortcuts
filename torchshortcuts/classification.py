import torch
import torch.nn as nn
import torch.nn.functional as F

from . import main_device


class MLPClassifier(nn.Module):
    """
    A fully connected multilayer perceptron with ReLU activations and LogSoftmax output
    """

    def __init__(self, input_dim, num_classes, hidden_layers=None, drop_p=0.2):
        """
        Constructs the MLP network

        :param input_dim: dimension of the input space
        :param num_classes: number of classes in the data (dimension of the output space)
        :param hidden_layers: list of sizes for the hidden layers
        :param drop_p: dropout rate
        """
        super().__init__()

        if hidden_layers is None:
            self.out = nn.Linear(input_dim, num_classes)
            self.hidden = []
        else:
            self.hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
            for i in range(1, len(hidden_layers)):
                self.hidden.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i])])
            self.out = nn.Linear(hidden_layers[-1], num_classes)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):

        for hd in self.hidden:
            x = F.relu(hd(x))
            x = self.dropout(x)

        x = F.log_softmax(self.out(x), dim=1)
        return x


def validation(model, testloader,
               criterion=nn.NLLLoss(), input_transform=None, output_str="test loss={:.3f}, test accuracy={:.3f}"):
    """
    Validates the model against a test set. This function will put the model in eval mode.

    :param model: the NN
    :param testloader: test set
    :param criterion: the criterion to compute the loss (it should be NNNLoss)
    :param input_transform: a function that should be applied befeore feeding the input to the NN
    :param output_str: string format of the output, should have two replacement fields

    :return: (loss, accuracy) over the test set
    """
    accuracy = 0
    test_loss = 0

    model.eval()
    model.to(main_device)

    with torch.no_grad():

        for features, labels in testloader:

            if input_transform is not None:
                features = input_transform(features)

            features, labels = features.to(main_device), labels.to(main_device)

            outputs = model(features)
            test_loss += criterion(outputs, labels).item()

            # accuracy
            ps = torch.exp(outputs)
            equality = labels.data == ps.argmax(dim=1)
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy, output_str.format(test_loss/len(testloader), accuracy/len(testloader))
