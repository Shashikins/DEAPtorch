from deap import base, creator, tools, algorithms
import numpy
import pickle
import multiprocessing
import multiprocessing.pool
import torch

from .helpers.setup import setup_toolbox, setup_creator
from .helpers.operators import register_operators
from .helpers.evaluation import eval_individual, init_worker, NewPool

def optimize_hyperparameters(hyperparam_space, eval_func, ngen=5, pop_size=10, parallelize=True, processes: int = None):
    
    """
    Optimizes the hyperparameters in the given dictionary using genetic algorithms with DEAP. 
    
    This function supports parallel execution, using multiple GPUs if available, or CPUs with specified parallel processes. The user can specify whether they wish to use this feature or not. If the user does wish to use parallelization, the user's evaluation function should be set up to handle this.
    
    The evaluation function provided by the user should be capable of receiving a dictionary of hyperparameters with specific values and returning a fitness score based on the model's performance. This fitness score is accuracy or another performance measure to be maximized. It should be a float in a single-value tuple.
    
    Since multiprocessing is used, the user's code should also protect the process pool in a if __name__ == "__main__" section, as seen in the examples provided.

    Parameters:
    - hyperparam_space (dict): A dictionary defining the hyperparameter names and their bounds. Expected format for each hyperparameter: {'name': (min_value, max_value)}. Each hyperparameter can have a (min_value, max_value) range using either ints or floats. 
    - eval_func (callable): A function that accepts a dictionary of hyperparameters and returns a tuple containing a single fitness score to be maximized.
    - ngen (int, optional): Number of generations to run the optimization for. Default is 5.
    - pop_size (int, optional): Number of individuals in each generation. Default is 10.
    - parallelize (bool, optional): Flag to enable parallel processing. Default is True.
    - processes (int, optional): Number of processes to use if parallelizing manually. Defaults to None, using all available GPUs or a single CPU.

    Returns:
    - dict: A dictionary containing the optimized hyperparameters.
    """
    multiprocessing.set_start_method('spawn', force=True)
    
    # Detect GPU availability and set the number of processes accordingly.
    if torch.cuda.is_available():
        if parallelize:
            num_processes = torch.cuda.device_count()
            print(f"Found {num_processes} GPUs.")
        else:
            num_processes = 1
    else:
        # Fallback to CPU, we don't parallelize with a CPU because PyTorch already uses OpenMP (if the user wants to override this and parallelize it anyway, they can specify the number of processes when calling the function)
        num_processes = 1  
        print(f"No GPUs found. Using CPU.")
    
    # Override the automatic detection with a user-specified number of processes if provided.
    if parallelize and processes is not None:
        num_processes = processes
 
    # Create a multiprocessing pool to manage parallel task execution.
    pool = NewPool(processes=num_processes)
    pool.map(init_worker, range(num_processes))
    
    # Initialize the genetic algorithm components with DEAP.
    hyperparam_names = list(hyperparam_space.keys())
    setup_creator()
    toolbox = setup_toolbox(hyperparam_space)
    register_operators(toolbox, hyperparam_space)
    toolbox.register("evaluate", eval_individual, eval_func=eval_func, hyperparam_names=hyperparam_names)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    # Setting up statistics^
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    # The following code is based on eaSimple from https://github.com/DEAP/deap/blob/master/deap/algorithms.py, but with parallelization added using pool.map
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluate the individuals with an invalid fitness in the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = pool.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    if hof is not None:
        hof.update(pop)

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
            
        # Select the next generation of individuals
        offspring = toolbox.select(pop, len(pop))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.2)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = pool.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    
    best_hyperparams = {name: val for name, val in zip(hyperparam_names, hof[0])}
    
    print(best_hyperparams)
    print(logbook)
    
    # Save the logbook to file for later analysis.
    with open("logbook.pkl", "wb") as lb_file:
        pickle.dump(logbook, lb_file)
    
    return best_hyperparams
