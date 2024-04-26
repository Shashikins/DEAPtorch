# DEAPtorch

DEAPtorch is a Python library that automates the optimization of hyperparameters for machine learning models. It does this by integrating the Distributed Evolutionary Algorithms in Python (DEAP) library for quick and easy use with PyTorch. 

## Features

- **Hyperparameter Optimization**: Automate the search for optimal machine learning model hyperparameters.
- **Parallel Processing Support**: Utilize multiple GPUs or CPU processes to optimize the training and testing of the model with different hyperparameters.

## Usage

DEAPtorch is designed to do the work of managing evolutionary algorithms on behalf of the user. To use this library, the user can simply:

- Import DEAPtorch:
`from deaptorch import optimize_hyperparameters`

- Create a dictionary for the hyperparameters you wish to optimize. This dictionary should define the hyperparameter names as keys and their bounds as values. Expected format for each hyperparameter: {'name': (min_value, max_value)}. Each hyperparameter can have a (min_value, max_value) range using either ints or floats. For example:
```
hyperparam_space = {
    'learning_rate': (0.0001, 0.1),
    'momentum': (0.0, 0.95),
    'epochs': (5, 20),
    'gamma': (0.1, 0.9),
    'batch_size': (50, 100),
    'test_batch_size': (500, 2000),
}
```

- Write a callable function that accepts a dictionary of optimized hyperparameters, trains and tests the machine learning model using these hyperparameters, and returns a corresponding performance metric. This performance metric should be a single value that needs to be maximized (for example, accuracy, or -loss). This single performance metric should be wrapped in a tuple when returned. For example, this could be the argument accepted by the function:
```
test_hyperparams = {
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 14,
    'learning_rate': 0.05,
    'gamma': 0.7
}
```
The callable function should then use these values to train a machine learning model and return its corresponding performance metric. For examples of such callable functions, see the examples folder. 
The return statement could then be, for example:
`return (accuracy,)`

- Since multiprocessing is used, the user's code should protect the process pool in a if __name__ == "__main__" section. For example:
```
if __name__ == '__main__':
    main()
```

- If the user wishes to use parallelization of the training and testing of the machine learning models, this must be supported in their callable function. 
If the parallelize flag is set to False, the user doesn't need to do this because parallelization will not be enabled. 
If the parallelize flag is set to True, as it is by default, DEAPtorch will check whether PyTorch detects any available GPUs.
If only a CPU is available, by default the number of processes is set to 1 so as not to interfere with how OpenMP is already used by PyTorch. If they want, the user can however override this by specifying the number of processes they wish to use.
If a GPU is available, DEAPtorch will distribute its calls to the callable function across these GPUs using multiprocessing so that the training and testing of the machine learning models can be done in parallel.
In any case, the rank can be retrieved through the environment variable CUDA_VISIBLE_DEVICES.

- The user can then use the library to optimize their hyperparameters, by calling the optimize_hyperparameters function of the library. See the "optimize_hyperparameters" section of this README. For example:
`best_hyperparams = optimize_hyperparameters(hyperparam_space, train_and_evaluate, ngen=8, pop_size=15)`

## optimize_hyperparameters

Optimizes the hyperparameters in the given dictionary using genetic algorithms with DEAP. 

This function supports parallel execution, using multiple GPUs if available, or CPUs with specified parallel processes. The user can specify whether they wish to use this feature or not. If the user does wish to use parallelization, the user's evaluation function should be set up to handle this.

The evaluation function provided by the user should be capable of receiving a dictionary of hyperparameters with specific values and returning a fitness score based on the model's performance. This fitness score is accuracy or another performance measure to be maximized. It should be a float in a single-value tuple.

Since multiprocessing is used, the user's code should also protect the process pool in a if __name__ == "__main__" section, as seen in the examples provided.

Parameters:
- hyperparam_space (dict): A dictionary defining the hyperparameter names and their bounds. Expected format for each hyperparameter: {'name': (min_value, max_value)}. Each hyperparameter can have a (min_value, max_value) range using either ints or floats. 
- eval_func (callable): A function that accepts a dictionary of hyperparameter names and values. Expected format for each hyperparameter: {'name': value}. It should return a tuple containing a single fitness value to be maximized.
- ngen (int, optional): Number of generations to run the optimization for. Default is 5.
- pop_size (int, optional): Number of individuals in each generation. Default is 10.
- parallelize (bool, optional): Flag to enable parallel processing. If True and GPUs are available, it will use GPU parallel processing; otherwise, it will fall back to using CPUs. Default is True.
- processes (int, optional): Number of processes to use if parallelizing manually. This parameter is considered only if parallelize is True. Defaults to None, using all available GPUs or a single CPU.

Returns:
- dict: A dictionary containing the optimized hyperparameters.

## Licence

This project is licensed under the GNU Lesser General Public License, which can be found here: https://www.gnu.org/licenses/lgpl-3.0.en.html
