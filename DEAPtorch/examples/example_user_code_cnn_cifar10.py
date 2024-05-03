# CIFAR 10 with a CNN
# code adapted from https://github.com/pytorch/examples/tree/main/mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# import from DEAPtorch
from deaptorch import optimize_hyperparameters
import math
import time

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    # Forward pass through the network
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Training function for the model
def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Testing function for the model
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)
    return (accuracy,) #must be a tuple

# Set up callable function
def train_and_evaluate(best_hyperparams):
    
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    # set seed to 1
    torch.manual_seed(1)

    # Set device based on CUDA availability
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(device)

    train_kwargs = {'batch_size': best_hyperparams['batch_size']}
    test_kwargs = {'batch_size': best_hyperparams['test_batch_size']} 
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    dataset1 = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=best_hyperparams['learning_rate'], momentum=best_hyperparams['momentum'])

    scheduler = StepLR(optimizer, step_size=1, gamma=best_hyperparams['gamma'])

    # Training and testing loop
    for epoch in range(1, best_hyperparams['epochs'] + 1):
        train(model, device, train_loader, optimizer, epoch)  #add log_interval here if different from 10
        performance = test(model, device, test_loader)
        scheduler.step()

    # Return single performance metric to maximize, in a tuple
    return performance 

# Set up hyperparameters with ranges
hyperparam_space = {
    'learning_rate': (0.0001, 0.1), # Range of floats
    'momentum': (0.0, 0.95),
    'epochs': (5, 20), # Range of ints
    'gamma': (0.1, 0.9),
    'batch_size': (50, 100),
    'test_batch_size': (500, 2000),
}

def main():
    
    start_time = time.time()

    # use DEAPtorch
    best_hyperparams = optimize_hyperparameters(hyperparam_space, train_and_evaluate, ngen=5, pop_size=10)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution took: {elapsed_time} seconds")
    print(best_hyperparams)

# Protect the entry point of the program
if __name__ == '__main__':
    main()
