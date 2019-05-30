"""
Implements the Swendsen-Wang (SW) cluster update MC algorithm to generate
samples of the spin-1/2 Ising Model.

NOTES:
- This is implemented in numpy (and does not make use of TensorFlow). Thus,
samples must be generated before running the results through TF's compute graphs.

Swendsen-Wang algorithm:
- Uses a Breadth-First Search (BFS) to create clusters of spins.
- If two neighboring sites have the same spin value, they probabilistically
    join the same cluster.
- Once all spins have been assigned to clusters, each cluster is fipped with
    probability 1/2 (independently of other clusters).

This algorithm is, thus, rejection-free, and converges much faster near the
critical temperature than single spin update techniques (e.g. Metropolis-Hastings)

TODO:
- Eliminate recursive-approach (this leads to recursion-depth errors for large
lattice systems, especially in the low temperature limit))
- We may consider trying to implement SW with TF (for easier debugging and
to allow TF to also optimize these graphs).


Inspired by:
Zhaonat: https://github.com/zhaonat/cluster_monte_carlo/blob/master/swendsen_wang_cluster/run_cluster.py

"""
import math, sys, collections, functools
from typing import List, Callable, NamedTuple, Tuple
import _pickle as pickle
from tqdm import tqdm

import numpy as np

from rsmi_renormalization import visualize

#np.set_printoptions(threshold=sys.maxsize)

#------------------------------------------------------------------------
#
# Base Lattice Kernel
#
#------------------------------------------------------------------------


class BaseLatticeSW(object):
    def __init__(self,
                 lattice_width: int,
                 name="lattice_kernel"):
        self.lattice_width = lattice_width
        self.n_spins = self.lattice_width** 2 # TODO generalize to d dimensiosn
        self._name = name


    def one_step(self, current_state, debug=False, log=False):

        """Progress one step for each chain.

        Performs the Swendsen-Wang algorithm.
            "Bond variables" b[i,j] for pairs of neighboring spins (i, j)
                s_i != s_j => b[i,j] = 0
                s_i == s_j => b[i,j] = 1 w/ prob. 1-exp(-2 * J * s_i * s_j)
                                     = 0 otherwise.

            Bond variables form adjacency matrix from which we can determine
                uncorrelated clusters of spins. These we flip, individually,
                with probability 1/2

            This is 'rejection-free'

        Practically, we employ a Breadth-First Search algorithm.
            Iterates over unvisited spins from 0 -> `n_spins`
            A given unvisited spin becomes the root of a new cluster.
            Decides to flip cluster with probability 1/2. Flips root node.
            Recursively visits children, applying cluster's flip rule.

        Args:
            current_state: shape [n_spins]
            previous_kernel_results: shape [1]

        """

        # A listof indices of elements of all clusters to be flipped
        clusters = {}
        considered = np.zeros(self.n_spins) # keeps track of which indices we have considered for a given cluster so we don't revisit already considered nodes
        visited = np.zeros(self.n_spins) # keeps track of which indices have been claimed

        def connect(root_idx, curr_idx):
            # Adds new elements to cluster
            nonlocal current_state
            nonlocal visited
            nonlocal considered

            considered[curr_idx] = root_idx
            neighbors_idxs = np.array(self._get_neighbor_idxs(curr_idx), dtype=int)

            # Activates with P 1-e^(-2J)
            # 1 + s_i s_j = {2 if s_i = s_j \\ 0 otherwise.}
            activations = np.random.uniform(size=[4])
            thresholds = np.exp(-self.J* (1. +(current_state[curr_idx] * current_state[neighbors_idxs])))
            bonded = activations > thresholds

            # Connect neighbors
            for i in neighbors_idxs[bonded]:
                # Probably a smarter np way to do this
                if visited[i] == 0 and considered[i] != root_idx and i not in clusters[root_idx]:
                    clusters[root_idx].append(i)
                    visited[i] = 1

            return curr_idx

        def build_cluster(root_idx):
            clusters[root_idx] = [root_idx]

            # Do you visit each node only once
            # or can you consider a node multiple times as in the Wolff Algorithm?
            i = 0
            while i < len(clusters[root_idx]):
                connect(root_idx, clusters[root_idx][i])
                i += 1

        # Build cluster, if i reaches len(cluster) it means no new elements have
        # been added in the last update and the algorithm terminates
        i = 0
        while i in range(len(visited)):
            if visited[i] == 0:
                build_cluster(i)

                if np.random.uniform() > 0.5:
                    current_state[clusters[i]] *= -1

            i += 1

        return current_state


#------------------------------------------------------------------------
#
# Ising Kernel
#
#------------------------------------------------------------------------

class IsingSW(BaseLatticeSW):
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

        BaseLatticeSW.__init__(self, lattice_width, name)


#------------------------------------------------------------------------
#
# Ising 2D Kernel(s) - Helical, Toroidal
#
#------------------------------------------------------------------------

class Ising2DHelicalSW(IsingSW):
    def __init__(self,
                 h,
                 J,
                 lattice_width,
                 name='ising_sampling_kernel') -> None:
        IsingSW.__init__(self, h, J, lattice_width, name)

        self._get_neighbor_idxs = lambda i: [(i - 1) % self.n_spins,
                                             (i + 1) % self.n_spins,
                                             (i - self.lattice_width) % self.n_spins,
                                             (i + self.lattice_width) % self.n_spins]

class Ising2DToroidalSW(IsingSW):
    def __init__(self,
                 h,
                 J,
                 lattice_width,
                 name='ising_sampling_kernel') -> None:
        IsingSW.__init__(self, h, J, lattice_width, name)

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

def ising_generator_sw(
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
        n_burnin_steps = 5

    if n_steps_between_results is None:
        n_steps_between_results = 5

    ising_kernel = Ising2DHelicalSW(
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


class BaseLatticeSWOG(object):
    def __init__(self,
                 lattice_width: int,
                 name="lattice_kernel"):
        self.lattice_width = lattice_width
        self.n_spins = self.lattice_width** 2 # TODO generalize to d dimensiosn
        self._name = name

    def one_step(self, current_state, debug=False, log=False):

        """Progress one step for each chain.

        Performs the Swendsen-Wang algorithm.
            "Bond variables" b[i,j] for pairs of neighboring spins (i, j)
                s_i != s_j => b[i,j] = 0
                s_i == s_j => b[i,j] = 1 w/ prob. 1-exp(-2 * J * s_i * s_j)
                                     = 0 otherwise.

            Bond variables form adjacency matrix from which we can determine
                uncorrelated clusters of spins. These we flip, individually,
                with probability 1/2

            This is 'rejection-free'

        Practically, we employ a Breadth-First Search algorithm.
            Iterates over unvisited spins from 0 -> `n_spins`
            A given unvisited spin becomes the root of a new cluster.
            Decides to flip cluster with probability 1/2. Flips root node.
            Recursively visits children, applying cluster's flip rule.

        Args:
            current_state: shape [n_spins]
            previous_kernel_results: shape [1]

        """

        # Stores the index of the root of the cluster. Of same shape as current_state
        visited = -np.ones([self.n_spins])

        def connect(curr_idx, root_idx, is_flipped):
            """ Performs recursive updating """

            nonlocal visited
            nonlocal current_state

            # Register visit with root_idx
            visited[curr_idx] = root_idx

            neighbors_idxs = np.array(self._get_neighbor_idxs(curr_idx), dtype=int)
            # Activates with P 1-e^(-2J)
            # 1 + s_i s_j = {2 if s_i = s_j \\ 0 otherwise.}
            activations = np.random.uniform(size=[4])
            thresholds = np.exp(-self.J* (1. +(current_state[curr_idx] * current_state[neighbors_idxs])))
            bonded = activations > thresholds
            accepted = np.logical_and(bonded, -1 == visited[neighbors_idxs])

            # Connect neighbors
            for i in neighbors_idxs[accepted]:
                connect(i, root_idx, is_flipped)

            if is_flipped:
                current_state[curr_idx] *= (1 - 2 * is_flipped)

            return curr_idx

        def root_init(root_idx):
            # Initialize the cluster root node and decide whether to flip that cluster
            is_flipped = np.random.uniform() > 0.5

            # Run the BFS
            connect(root_idx, root_idx, is_flipped)

        # Generate the clusters randomly
        i = np.random.randint(self.n_spins)
        while np.any(visited == -1):
            if visited[i] == -1:
                # Build a cluster and consider new random starting point
                root_init(i)
                i = np.random.randint(self.n_spins)

            # if current point is occupied, consider the next point (will still be random)
            i = (i + 1) % self.n_spins

        if self.J > 2:
            print(visited)

        return current_state
