from .wolff import *
from .swendsen_wang import *
from .metropolis_hastings import *
from .metropolis_hastings_tf import *
from .expectations import *

def ising_generator(mcmc, *args, **kwargs):
    if mcmc == "mh":
        return ising_generator_mh(*args, **kwargs)
    elif mcmc == "sw":
        return ising_generator_sw(*args, **kwargs)
    elif mcmc == "wf":
        return ising_generator_wf(*args, **kwargs)
