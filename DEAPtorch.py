from deap import base, creator, tools, algorithms
import random
import numpy
import matplotlib.pyplot as plt
import pickle
import multiprocessing
import os
import torch



def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def setup_toolbox(hyperparam_space):
    toolbox = base.Toolbox()
    for hp_name, (hp_min, hp_max) in hyperparam_space.items():
        #check on whether the bounds are floats or ints
        if isinstance(hp_min, int) and isinstance(hp_max, int):
            toolbox.register(f"attr_{hp_name}", random.randint, hp_min, hp_max)
        else:
            toolbox.register(f"attr_{hp_name}", random.uniform, hp_min, hp_max)
    def create_individual():
        individual = [toolbox.__getattribute__(f"attr_{hp_name}")() for hp_name in hyperparam_space]
        return individual
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def checkBounds(hyperparam_space):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i, name in enumerate(hyperparam_space):
                    min_val, max_val = hyperparam_space[name]
                    #if it's supposed to be an int, round it
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        child[i] = int(round(child[i]))
                    #keep the value in the bounds
                    child[i] = max(min(child[i], max_val), min_val)
            return offspring
        return wrapper
    return decorator
    
def register_operators(toolbox, hyperparam_space):
    #get bounds for each hyperparameter
    low_bounds, high_bounds = zip(*[(hp_details[0], hp_details[1]) for hp_details in hyperparam_space.values()])
    
    #mutation operator: choose based on hyperparameter type (float or int)
    for hp_name, (hp_min, hp_max) in hyperparam_space.items():
        #for ints
        if isinstance(hp_min, int) and isinstance(hp_max, int):
            toolbox.register(f"mutate_{hp_name}", tools.mutUniformInt, low=hp_min, up=hp_max, indpb=0.2)
        #for floats
        else:
            toolbox.register(f"mutate_{hp_name}", tools.mutPolynomialBounded, low=low_bounds, up=high_bounds, eta=1.0, indpb=0.2)

    def mutate(individual):
        for i, name in enumerate(hyperparam_space.items()):
            if hasattr(toolbox, f"mutate_{name}"):
                toolbox.__getattribute__(f"mutate_{name}")(individual)
        return individual,

    toolbox.register("mutate", mutate)

    #works for both ints and floats
    toolbox.register("mate", tools.cxOnePoint)

    #apply bounds check to all hyperparameters after crossover and mutation
    toolbox.decorate("mate", checkBounds(hyperparam_space))
    toolbox.decorate("mutate", checkBounds(hyperparam_space))
    toolbox.register("select", tools.selTournament, tournsize=3)
    
def eval_individual_with_gpu(individual, eval_func, hyperparam_names, num_processes):
    index = os.getpid() % num_gpus if num_gpus > 0 else 0
    hyperparams = {name: val for name, val in zip(hyperparam_names, individual)}
    #set CUDA_VISIBLE_DEVICES to the specified GPU index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    performance = eval_func(hyperparams)
    return performance

def eval_individual(individual, eval_func, hyperparam_names):
    hyperparams = {name: val for name, val in zip(hyperparam_names, individual)}
    performance = eval_func(hyperparams)
    return performance

def optimize_hyperparameters(hyperparam_space, eval_func, ngen=5, pop_size=10):
    """
    Processes the hyperparameters in the given dictionary. 

    Parameters:
    - hyperparam_space: A dictionary of hyperparameters.
    - train_and_evaluate A function that trains the model with the hyperparameters, and returns accuracy as a tuple.

    Returns:
    - A dictionary with optimized hyperparameters.
    """
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")
        num_processes = num_gpus
    else:
        num_processes = multiprocessing.cpu_count()  # Fallback to CPU
        print(f"No GPUs found. Using CPU with {num_processes} processes.")
 
    pool = multiprocessing.Pool(processes=num_processes)
    
    hyperparam_names = list(hyperparam_space.keys())
    
    setup_creator()
    toolbox = setup_toolbox(hyperparam_space)
    register_operators(toolbox, hyperparam_space)
    toolbox.register("evaluate", eval_individual_with_gpu, eval_func=eval_func, hyperparam_names=hyperparam_names, num_processes=num_processes)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    # Based on eaSimple from https://github.com/DEAP/deap/blob/master/deap/algorithms.py, but with parallelization added
    
    #_, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=True)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluate the individuals with an invalid fitness
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
        ##
        eval_jobs = []
        for ind in pop:
            # Assign each evaluation task to a different GPU process
            eval_jobs.append(pool.apply_async(eval_individual_with_gpu, (ind, eval_func, hyperparam_names, len(eval_jobs) % num_processes)))

        for job in eval_jobs:
            job.wait()

        # Get the results from the evaluation jobs
        fitnesses = [job.get() for job in eval_jobs]

        # Update individual fitness values
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        ##
            
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
    
    with open("logbook.pkl", "wb") as lb_file:
        pickle.dump(logbook, lb_file)
    
    return best_hyperparams
