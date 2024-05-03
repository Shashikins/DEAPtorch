from deap import tools
import random

def checkBounds(hyperparam_space):
    """
    Decorator to make sure the hyperparameter values stay inside in the given bounds.

    Is also used to make sure that ints remain ints after mutation and crossover.

    Parameters:
    - hyperparam_space (dict): A dictionary defining the hyperparameter names and their bounds. Expected format for each hyperparameter: {'name': (min_value, max_value)}. Each hyperparameter can have a (min_value, max_value) range using either ints or floats. 

    Returns:
    - function: A decorator function to wrap genetic operators.
    """
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i, name in enumerate(hyperparam_space):
                    min_val, max_val = hyperparam_space[name]
                    # If the value is supposed to be an int, round it to an int
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        child[i] = int(round(child[i]))
                    # Keep the value in the bounds
                    child[i] = max(min(child[i], max_val), min_val)
            return offspring
        return wrapper
    return decorator
    
def register_operators(toolbox, hyperparam_space):
    """
    Register genetic operators in the DEAP toolbox based on the hyperparameter space.

    Parameters:
    - toolbox (deap.base.Toolbox): The DEAP toolbox to configure.
    - hyperparam_space (dict): A dictionary defining the hyperparameter names and their bounds. Expected format for each hyperparameter: {'name': (min_value, max_value)}. Each hyperparameter can have a (min_value, max_value) range using either ints or floats. 
    """
    # Get bounds for each hyperparameter
    low_bounds, high_bounds = zip(*[(hp_details[0], hp_details[1]) for hp_details in hyperparam_space.values()])
    
    # Mutation operator: choose based on hyperparameter type (float or int)
    for hp_name, (hp_min, hp_max) in hyperparam_space.items():
        # For ints
        if isinstance(hp_min, int) and isinstance(hp_max, int):
            toolbox.register(f"mutate_{hp_name}", tools.mutUniformInt, low=hp_min, up=hp_max, indpb=0.2)
        # For floats
        else:
            toolbox.register(f"mutate_{hp_name}", tools.mutPolynomialBounded, low=low_bounds, up=high_bounds, eta=1.0, indpb=0.2)

    # Composite mutation function that applies all registered mutation operators to an individual.
    def mutate(individual):
        """
        Mutates an individual by applying mutation operators. Iterates over each hyperparameter, applying the registered mutation operator for each hyperparameter.

        Parameters:
        - individual (list): The individual to mutate, represented as a list of values for hyperparameters.

        Returns:
        - tuple: The mutated individual wrapped in a tuple, which is required for DEAP.
        """
        for i, name in enumerate(hyperparam_space.items()):
            if hasattr(toolbox, f"mutate_{name}"):
                toolbox.__getattribute__(f"mutate_{name}")(individual)
        return individual,

    toolbox.register("mutate", mutate)

    # One-point crossover operator. Works for both ints and floats.
    toolbox.register("mate", tools.cxOnePoint)

    # Make sure all genetic operators respect the hyperparameter bounds.
    toolbox.decorate("mate", checkBounds(hyperparam_space))
    toolbox.decorate("mutate", checkBounds(hyperparam_space))
    
    # Tournament selection with a tournament size of three.
    toolbox.register("select", tools.selTournament, tournsize=3)
