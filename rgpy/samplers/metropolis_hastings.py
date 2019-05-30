"""
Implements the Metropolis-Hastings (MH) single-spin update MC algorithm to generate
samples of the spin-1/2 Ising Model.

NOTES:
- This is implemented in numpy (and does not make use of TensorFlow). Thus,
samples must be generated before running the results through TF's compute graphs.

"""
import math, sys, collections, functools
from typing import List, Callable, NamedTuple, Tuple
import _pickle as pickle
from tqdm import tqdm

import numpy as np

from rgpy import visualize

#np.set_printoptions(threshold=sys.maxsize)


#------------------------------------------------------------------------
#
# Base Lattice Kernel
#
#------------------------------------------------------------------------

class BaseLatticeMH(object):
    def __init__(self,
                 lattice_width: int,
                 name="lattice_kernel"):
        self.lattice_width = lattice_width
        self.n_spins = self.lattice_width** 2 # TODO generalize to d dimensiosn
        self._name = name

    def one_step(self, current_state, debug=False, log=False):

        """Progress one step for each chain.

        Args:
            current_state: shape [n_spins]
            previous_kernel_results: shape [1]

        """

        # Stores the index of the root of the cluster. Of same shape as current_state

        # NOT YET IMPLEMENTED

        i = np.random.randint(self.n_spins)
        s_i = current_state[i]

        i_nn = self._get_neighbor_idxs(i)
        nn = current_state[i_nn]

        delta_energy = -2. * self.J * s_i * np.sum(nn)
        activation = np.random.uniform()
        threshold = math.e ** (delta_energy)
        is_flipped = activation < threshold

        return current_state

#------------------------------------------------------------------------
#
# Ising Kernel
#
#------------------------------------------------------------------------

class IsingMH(BaseLatticeMH):
    """Makes a transition kernel that does sampling of an Ising model

    Args:
      samplers: A list of samplers that take a slice of state and return a
        distribution.
      target_log_prob_fn: A function to compute the log probability of state.
    """
    def __init__(self,
                 h,
                 J,
                 lattice_width:int,
                 name='ising_sampling_kernel') -> None:
        self.h = h
        self.J = J

        BaseLatticeMH.__init__(self, lattice_width, name)


#------------------------------------------------------------------------
#
# Ising 2D Kernel(s) - Helical, Toroidal
#
#------------------------------------------------------------------------

class Ising2DHelicalMH(IsingMH):
    def __init__(self,
                 h,
                 J,
                 lattice_width,
                 name='ising_sampling_kernel') -> None:
        IsingMH.__init__(self, h, J, lattice_width, name)

        self._get_neighbor_idxs = lambda i: [(i - 1) % self.n_spins,
                                             (i + 1) % self.n_spins,
                                             (i - self.lattice_width) % self.n_spins,
                                             (i + self.lattice_width) % self.n_spins]

class Ising2DToroidalMH(IsingMH):
    def __init__(self,
                 h,
                 J,
                 lattice_width,
                 name='ising_sampling_kernel') -> None:
        IsingMH.__init__(self, h, J, lattice_width, name)

        def _get_neighbor_idxs(I):
            i = I // self.lattice_width
            j = I % self.lattice_width
            ij_to_I = lambda i, j: i * self.lattice_width + j
            neighbors_ij = [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)]
            return [ij_to_I(*n_ij) % self.n_spins for n_ij in neighbors_ij]

        self._get_neighbor_idxs = _get_neighbor_idxs

    def _local_field(self, current_state, I):
        return tf.reduce_sum(tf.gather_nd(current_state,
                                          neighbors_I))

def ising_generator_mh(
        lattice_width=8,
        J=0.4406,
        h=0, # Currently makes no difference
        n_results_per_chain=10,
        n_chains=10,
        n_burnin_steps=None,
        n_steps_between_results=None):
    """
    n_results = `n_results_per_chain` * `n_chains`
    n_spins = `lattice_width` ** 2
    Generates n_results samples of an Ising system of `lattice_width` at `J`.
        (helical boundary conditions)

    Note:
        This is a wrapper for tfp.mcmc.sample_chain, with modifications:
            Arguments specific to Ising 2d (lattice_width, J)
            No arguments `chain_results` and `previous_kernel_results`
                (each chain starts from a randomly initialized state)
            This returns all results accumulated along the first axis
        Currently does nothing with kernel_results. This may change.

    Args:
        `lattice_width`: the width of the lattice
        `J`: nearest-neighbor coupling constant
        `H`: external magnetic field constant
        `n_results_per_chain`: number of results to generate per chain
        `n_chain`: number of chains to run (in parallel).
            Analagous to `parallel_iterations` in tfp.mcmc.sample_chain
        `n_burnin_steps`: number of steps to let the system 'thermalize' before
            taking the first result. (Default= `n_spins` ** 2)
        `n_steps_between_results`: number of steps between results
            (to reduce correlated outcomes). (Default= `n_spins` ** 2)

    Returns:
        `results` (List[tf.Tensor]): newly-generated 2d ising samples
            shape [`n_results`, `n_spins`]
    """

    n_results = n_results_per_chain * n_chains
    n_spins = lattice_width ** 2

    # TODO: more research into optimal #s for below to avoid autocorrelation
    if n_burnin_steps is None:
        n_burnin_steps = n_spins

    if n_steps_between_results is None:
        n_steps_between_results = n_spins

    ising_kernel = Ising2DHelicalMH(
        J=J,
        h=h,
        lattice_width=lattice_width
    )
    #Initialize results matrix
    samples = np.zeros([n_results, n_spins])

    # Burn-in
    tmp = np.random.choice(a=[-1., 1.], size=(n_spins))
    for i in range(n_burnin_steps):
        tmp = ising_kernel.one_step(tmp, debug=False, log=False)


    # MCMC iterations
    samples[0] = tmp

    print("Generating Samples")
    for i in tqdm(range(1, n_results * n_steps_between_results)):
        tmp = ising_kernel.one_step(tmp, log=False)
        if i % n_steps_between_results == 0:
            samples[i // n_steps_between_results] = tmp

    return samples
