# Time sequence prediction - use an LSTM to learn Sine waves
# code adapted from https://github.com/pytorch/examples/tree/main/time_sequence_prediction
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import from DEAPtorch
from deaptorch import optimize_hyperparameters

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        # Send to device
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

def generate_data():
    np.random.seed(2) 
    T = 20
    L = 1000
    N = 100
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    return data

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Set up callable function
def train_and_evaluate(hyperparams):
    # Set device based on CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # set random seed to 0
    torch.manual_seed(0)
    
    # Initialize model and move it to the device
    seq = Sequence().double().to(device)
    criterion = nn.MSELoss()
    
    data = generate_data()
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    # Move data to the device
    input = input.to(device)
    target = target.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    # Set hyperparameters
    optimizer = optim.LBFGS(seq.parameters(), lr=hyperparams['learning_rate'])
    for i in range(hyperparams['steps']):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        
    with torch.no_grad():
        future = 1000
        pred = seq(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.item())
        rmse_val = rmse(pred[:, :-future].cpu().numpy(), test_target.cpu().numpy())
        print('test RMSE:', rmse_val)
    
    # Return single performance metric to maximize, in a tuple
    return (-1*rmse_val,)

if __name__ == '__main__':
    
    # Set up hyperparameters with ranges
    hyperparam_space = {
        'learning_rate': (0.0001, 0.1), # Range of floats
        'steps': (5, 20) # Range of ints
    }
        
    best_hyperparams = optimize_hyperparameters(hyperparam_space, train_and_evaluate, ngen=3, pop_size=5)
    print('Best hyperparameters:', best_hyperparams)
    
    
