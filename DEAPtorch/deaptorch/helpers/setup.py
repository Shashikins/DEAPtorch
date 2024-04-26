from deap import base, creator, tools
import random

def setup_creator():
    """
    Sets up the DEAP creator for defining fitness and individual classes.
    
    This function defines a maximization fitness measure ('FitnessMax') and an 'Individual' class which will be used to store a list of hyperparameter values along with a fitness value. 
    """
    # 'FitnessMax' is for maximizing a single-objective fitness. This defines how individuals are assessed.
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # 'Individual' is defined as a list with a fitness attribute
    creator.create("Individual", list, fitness=creator.FitnessMax)

def setup_toolbox(hyperparam_space):
    """
    Configures the DEAP toolbox.
    
    Each hyperparameter in the `hyperparam_space` dictionary is used to register functions in the toolbox that generate random values within specified bounds. This setup is used for initializing the population with varied individuals.
    
    This function handles hyperparameters with ranges that can be either ints or floats.

    Parameters:
    - hyperparam_space (dict): A dictionary defining the hyperparameter names and their bounds. Expected format for each hyperparameter: {'name': (min_value, max_value)}. Each hyperparameter can have a (min_value, max_value) range using either ints or floats. 

    Returns:
    - toolbox (deap.base.Toolbox): The configured DEAP toolbox with methods for creating individuals and populations.
    """
    toolbox = base.Toolbox()
    
    # Register the attribute generator functions for each hyperparameter.
    for hp_name, (hp_min, hp_max) in hyperparam_space.items():
        # Check whether the bounds are floats or ints
        if isinstance(hp_min, int) and isinstance(hp_max, int):
            toolbox.register(f"attr_{hp_name}", random.randint, hp_min, hp_max)
        else:
            toolbox.register(f"attr_{hp_name}", random.uniform, hp_min, hp_max)
    
    # Function to create a single individual.
    def create_individual():
        """ 
        Generate an individual instance with random attributes based on the hyperparameter space. 
        """
        individual = [toolbox.__getattribute__(f"attr_{hp_name}")() for hp_name in hyperparam_space]
        return individual
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    return toolbox
