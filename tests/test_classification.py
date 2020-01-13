from .context import torchshortcuts

from torchshortcuts.classification import MLPClassifier


model = MLPClassifier(28*28, 10, [512, 256])
print(model)
