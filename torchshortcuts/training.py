from . import main_device


def train(model, trainloader, testloader, criterion, optimizer, epochs,
          validation=None, input_transform=None, print_every=None):
    """
    Trains a generic neural network

    :param model: the NN
    :param trainloader: training set
    :param testloader: test set
    :param criterion: loss function
    :param optimizer: optimization method
    :param epochs: number of ephocs for training
    :param validation: function to compute validation on the training set
    :param input_transform: a function that should be applied befeore feeding the input to the NN
    :param print_every: number of batches before logging (default=1 epoch)

    :return: the trained model
    """

    if validation is None:
        raise RuntimeError("validation function cannot be None")

    steps = 0

    model.to(main_device)
    model.train()

    training_loss = 0

    if print_every is None:
        print_every = len(trainloader)

    for e in range(epochs):

        for features, labels in trainloader:

            steps += 1

            if input_transform is not None:
                features = input_transform(features)

            features, labels = features.to(main_device), labels.to(main_device)

            optimizer.zero_grad()

            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if steps % print_every == 0:

                _, _, test_str = validation(model, testloader, criterion=criterion, input_transform=input_transform)

                print("Epoch {}/{}: ".format(e+1, epochs),
                      "test loss={:.3f}".format(training_loss/print_every),
                      test_str)

                model.train()

                training_loss = 0

    return model
