import multiprocessing
import multiprocessing.pool
import os

def eval_individual(individual, eval_func, hyperparam_names):
    """
    Evaluates an individual by applying the user's evaluation function with the hyperparameters.

    Parameters:
    - individual (list): The individual from the population, represented as a list of hyperparameter values.
    - eval_func (callable): The evaluation function provided by the user, which should accept a dictionary of hyperparameters and return accuracy or another perfomance measure to maximize.
    - hyperparam_names (list): List of the hyperparameter names corresponding to the values in the individual.

    Returns:
    - tuple: A single-value tuple containing a float, which is the fitness score of the individual. DEAP requires this to be a tuple.
    """
    hyperparams = {name: val for name, val in zip(hyperparam_names, individual)}
    performance = eval_func(hyperparams)
    return performance

def init_worker(index):
    """
    Initialize a worker process, setting the CUDA_VISIBLE_DEVICES environment variable to restrict GPU visibility.

    Parameters:
    - index (int): Index of the GPU to use for this process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    
# To avoid errors because PyTorch also uses multiprocessing
class NoDaemonProcess(multiprocessing.Process):
    """
    A non-daemon multiprocessing.Process.

    Overrides the default daemon properties of multiprocessing.Process to allow subprocesses to spawn children. 
    This is necessary because the default Process class does not allow the creation of subprocesses within a daemon.
    """
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
    
class NewPool(multiprocessing.pool.Pool):
    """
    Custom multiprocessing pool that uses NoDaemonProcess.

    This class modifies the process creation to use NoDaemonProcess, which allows each process in the pool to spawn subprocesses.
    """
    @staticmethod
    def Process(ctx, *args, **kwargs):
        return NoDaemonProcess(*args, **kwargs)
