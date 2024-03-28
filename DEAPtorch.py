from deap import base, creator, tools, algorithms
import random
import numpy

def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def setup_toolbox(learning_rate_min, learning_rate_max):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, learning_rate_min, learning_rate_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def register_operators(toolbox):
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.001, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

def eval_individual(individual, eval_func):
    learning_rate = individual[0]
    performance = eval_func({'learning_rate': learning_rate})
    return performance

def optimize_hyperparameters(hyperparam_space, eval_func, ngen=2, pop_size=2):
    """
    Processes the hyperparameters in the given dictionary. For each parameter,
    it applies specific processing rules. For example, for 'learning_rate',
    it sets the value to the average if it's given as a tuple.

    Parameters:
    - hyperparam_space: A dictionary of hyperparameters.
    - train_and_evaluate A function that trains the model with the hyperparameters, and returns accuracy as a tuple.

    Returns:
    - A dictionary with processed hyperparameters.
    """
    learning_rate_min = hyperparam_space['learning_rate'][0]
    learning_rate_max = hyperparam_space['learning_rate'][1]
    
    setup_creator()
    toolbox = setup_toolbox(learning_rate_min, learning_rate_max)
    register_operators(toolbox)
    toolbox.register("evaluate", eval_individual, eval_func=eval_func)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

    return hof[0][0]  # Best learning rate
