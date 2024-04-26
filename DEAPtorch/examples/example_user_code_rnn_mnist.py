# MNIST with an RNN
# code adapted from https://github.com/pytorch/examples/blob/main/mnist_rnn/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
# Import from DEAPtorch
from deaptorch import optimize_hyperparameters

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, input):
        # Shape of input is (batch_size,1, 28, 28)
        # converting shape of input to (batch_size, 28, 28)
        # as required by RNN when batch_first is set True
        input = input.reshape(-1, 28, 28)
        output, hidden = self.rnn(input)

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        output = output[:, -1, :]
        output = self.batchnorm(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        output = F.log_softmax(output, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, hyperparams):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % hyperparams['batch_size'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def train_and_evaluate(hyperparams):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    torch.manual_seed(1)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Apply hyperparameters
    batch_size = hyperparams['batch_size']
    test_batch_size = hyperparams['test_batch_size']
    epochs = hyperparams['epochs']
    lr = hyperparams['learning_rate']
    gamma = hyperparams['gamma']
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, hyperparams)
        test_accuracy = test(model, device, test_loader)
        scheduler.step()

    return (test_accuracy,)
    
def main():
    # Training settings
    
    #parser.add_argument('--batch-size', type=int, default=50, metavar='N',#64
                        #help='input batch size for training (default: 64)')#
    #parser.add_argument('--test-batch-size', type=int, default=800, metavar='N',1000
                        #help='input batch size for testing (default: 1000)') 
    #parser.add_argument('--epochs', type=int, default=14, metavar='N',#14
                        #help='number of epochs to train (default: 14)')
    #parser.add_argument('--lr', type=float, default=0.01, metavar='LR', #0.1
                        #help='learning rate (default: 0.1)')
    #parser.add_argument('--gamma', type=float, default=0.1, metavar='M', #0.7
                        #help='learning rate step gamma (default: 0.7)')

    hyperparam_space = {
        'batch_size': (50, 100),
        'test_batch_size': (500, 1000),
        'epochs': (10, 15),
        'learning_rate': (0.01, 0.1),
        'gamma': (0.1, 0.9),
    }

    best_hyperparams = optimize_hyperparameters(hyperparam_space, train_and_evaluate, ngen=2, pop_size=3)
    print("Best hyperparameters:", best_hyperparams)

if __name__ == '__main__':
    main()
