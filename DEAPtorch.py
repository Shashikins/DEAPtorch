from deap import base, creator, tools, algorithms
import random
import numpy
import matplotlib.pyplot as plt
import pickle

def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def setup_toolbox(hyperparam_space):
    toolbox = base.Toolbox()
    for hp_name, (hp_min, hp_max) in hyperparam_space.items():
        toolbox.register(f"attr_{hp_name}", random.uniform, hp_min, hp_max)
    def create_individual():
        return [toolbox.__getattribute__(f"attr_{hp_name}")() for hp_name in hyperparam_space]
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
                    range_val = max_val - min_val
                    #modulo operation to handle out-of-bounds with wrapping
                    if child[i] < min_val or child[i] > max_val:
                        child[i] = min_val + (child[i] - min_val) % range_val
            return offspring
        return wrapper
    return decorator

def register_operators(toolbox, hyperparam_space):
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    #apply bounds check to all hyperparameters
    hyperparam_bounds = {name: (hp_min, hp_max) for name, (hp_min, hp_max) in hyperparam_space.items()}
    toolbox.decorate("mutate", checkBounds(hyperparam_bounds))
    toolbox.register("select", tools.selTournament, tournsize=3)

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
    hyperparam_names = list(hyperparam_space.keys())
    
    setup_creator()
    toolbox = setup_toolbox(hyperparam_space)
    register_operators(toolbox, hyperparam_space)
    toolbox.register("evaluate", eval_individual, eval_func=eval_func, hyperparam_names=hyperparam_names)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    _, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=True)
    
    best_hyperparams = {name: val for name, val in zip(hyperparam_names, hof[0])}
    
    print(best_hyperparams)
    print(logbook)
    
    with open("logbook.pkl", "wb") as lb_file:
        pickle.dump(logbook, lb_file)
    
    return best_hyperparams  # Best learning rate
