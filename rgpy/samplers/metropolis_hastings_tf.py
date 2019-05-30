import collections
import functools
from typing import List, Callable, NamedTuple, Tuple

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from rsmi_renormalization import visualize
#------------------------------------------------------------------------
#
# Base Lattice Kernel
#
#------------------------------------------------------------------------

class BaseLatticeKernel(tfp.mcmc.TransitionKernel):
    def __init__(self,
                 n_spins: int,
                 name="lattice_kernel"):
        self.n_spins = n_spins
        self._name = name

        sign_flippers = np.ones((n_spins, n_spins))
        idxs = np.arange(n_spins)
        sign_flippers[idxs, idxs] = -1
        self.sign_flippers = tf.constant(sign_flippers, dtype=tf.float32)

    def get_delta_energy(self,
                         current_state: List[tf.Tensor],
                         current_energy: tf.Tensor,
                         prime_state: List[tf.Tensor], *args):
        """
        The energy change that would result from flipping spin i of
        current_state (which has total energy current_energy).

        Naive implementation calculates energy of the prime state

        Args:
            current_state: the immediate configuration of spins
            current_energy: the corresponding energy
            prime_state (List[tf.Tensor]): the configuration that would result
               from flipping spin i
            *args: further details about the flip
               (for child classes to provide more efficient implementations)

        Returns:
            delta_energy (tf.Tensor): change in energy for flip of spin i
            next_energy (tf.Tensor): energy of prime_state
        """
        next_energy = self._get_energy(prime_state)
        delta_energy = next_energy - current_energy
        return delta_energy, next_energy

    def get_energy(self, state):
        raise NotImplementedError

    def gen_possible_step(self, state):
        """ Chooses a random index of a spin to flip in state.

        Returns:
            i (tf.Tensor): index of spin
            prime_state (List[tf.Tensor]]): result of flipping spin i in state
        """
        i = tf.random.uniform([], minval=0, maxval=self.n_spins, dtype=tf.int32)
        sign_flipper = self.sign_flippers[i]
        prime_state = sign_flipper * state
        return i, prime_state

    def _one_step(self,
                  current_state: List[tf.Tensor],
                  previous_kernel_results: List[tf.Tensor]) -> Tuple[
                      List[tf.Tensor], List[tf.Tensor]]:

        """Progress one step for one chain.
        Each step only updates one element of state. Consider specifying
        `num_steps_between_results` in tfp.mcmc.sample_chain as len(samplers) - 1
        to obtain entirely new states for each result.

        Args:
            current_state: shape [n_spins]
            previous_kernel_results: shape [1]

        """
        # Previous kernel results contains the energy of the previous state
        current_energy = previous_kernel_results
        i, prime_state = self.gen_possible_step(current_state)
        delta_energy, next_energy = self.get_delta_energy(current_state,
                                                          current_energy,
                                                          prime_state,
                                                          i)
        def accept_flip():
            """
            Returns:
                prime_state: current state with spin i flipped
                next_energy:
            """
            nonlocal prime_state
            nonlocal next_energy
            #_prime_state = tf.Print(prime_state, [i], "Accepted flip: ")
            #_prime_state = tf.Print(_prime_state, [prime_state], 'New state: ', summarize=10)
            #_next_energy = tf.Print(next_energy, [delta_energy], 'With delta energy: ')

            return [prime_state, next_energy]

        def prob_flip():
            # Update state if randomly generated value in [0,1) exceeds the
            # relative probability
            accept_threshold = tf.exp(-delta_energy)
            accept_value = tf.random.uniform((1, ), dtype=tf.float32)[0]

            #accept_threshold = tf.Print(accept_threshold, [accept_threshold], "Threshold to accept: ")
            #accept_value = tf.Print(accept_value, [accept_value], "Random value: ")

            is_accepted = tf.greater_equal(accept_threshold, accept_value)
            #is_accepted = tf.Print(is_accepted, [is_accepted], "Probabilistic flip accepted? ")

            reject = current_state
            #reject = tf.Print(reject, [i], "Rejected flip: ")

            return tf.cond(
                is_accepted,
                accept_flip,
                lambda: [reject, current_energy])


        # if delta energy <= 0, accept the flip
        # else accept the flip with probability exp(-delta_energy)
        [next_state, next_energy] = tf.cond(
            tf.less_equal(delta_energy, tf.constant(0., dtype=tf.float32)),
            accept_flip,
            prob_flip)

        # Kernel results keep track of the energy of the previous configuration

        return next_state, next_energy

    def one_step(self, current_state: List[tf.Tensor],
                 previous_kernel_results: List[tf.Tensor]) -> Tuple[
                     List[tf.Tensor], List[tf.Tensor]]:

        """Progress one step for each chain.
        Each step only updates one element of state. Consider specifying
        `num_steps_between_results` in tfp.mcmc.sample_chain as len(samplers) - 1
        to obtain entirely new states for each result.

        Args:
            current_state: shape [n_chains, n_spins]
            previous_kernel_results: shape [n_chains, 1]

        """
        updates = self._one_step(current_state, previous_kernel_results)
        return updates

    def bootstrap_results(self, init_state: List[tf.Tensor]) -> List[tf.Tensor]:
        """Initiates results based off of initial state.
        Args:
          init_state: Initial state, usually specified in `current_state` of
            tfp.mcmc.sample_chain. shape [n_chains, n_spins]
        Returns:
          Initial accumulated results to begin the chain. shape [n_chains, 1]
        """
        return self.get_energy(init_state)

    @property
    def is_calibrated(self) -> bool:
        return True


#------------------------------------------------------------------------
#
# Generic Lattice Kernel
#
#------------------------------------------------------------------------

class GenericLatticeKernel(BaseLatticeKernel):
    def __init__(self,
                 n_spins: int,
                 energy_fn=lambda state: 0.,
                 name="rbm_sampling_kernel"):
        """Creates a kernel that can sample visible configurations for a trained
        RBM

        Args:
            energy_fn: function that returns the energy of a lattice state
        """
        self._get_energy = energy_fn
        BaseLatticeKernel.__init__(self, n_spins, name)

    def get_energy(self, state) -> tf.Tensor:
        return self._get_energy(state)

    def set_energy_fn(self,
                      energy_fn: Callable[[List[tf.Tensor]], tf.Tensor]) -> None:
        self._get_energy = energy_fn


def generic_generator_graph(
        energy_fn,
        n_spins=8,
        n_results_per_chain=10,
        n_chains=1,
        n_burnin_steps=None,
        n_steps_between_results=None,
        draw=True,
        save=True):
    """
    n_results = `n_results_per_chain` * `n_chains`

    Generates n_results samples of an generic lattice system of `n_spins`
        (helical boundary conditions)

    Note:
        This is a wrapper for tfp.mcmc.sample_chain, with modifications:
            No arguments `chain_results` and `previous_kernel_results`
                (each chain starts from a randomly initialized state)
            This returns all results accumulated along the first axis
        Currently does nothing with kernel_results. This may change.

    Args:
        `energy_fn`: the energy function
        `n_spins`: the number of units
        `n_results_per_chain`: number of results to generate per chain
        `n_chain`: number of chains to run (in parallel).
            Analagous to `parallel_iterations` in tfp.mcmc.sample_chain
        `n_burnin_steps`: number of steps to let the system 'thermalize' before
            taking the first result. (Default= `n_spins` ** 2)
        `n_steps_between_results`: number of steps between results
            (to reduce correlated outcomes). (Default= `n_spins` ** 2)
        `draw`: whether to draw the samples (to png) after generation.
            filename = 'J=<J>_h=<h>_lw=<lattice_width>.png'

    Yields:
        `results` (List[tf.Tensor]): newly-generated 2d ising samples
            shape [`n_results`, `n_spins`]
    """

    n_results = n_results_per_chain * n_chains

    # TODO: more research into optimal #s for below to avoid autocorrelation
    if n_burnin_steps is None:
        n_burnin_steps = n_spins

    if n_steps_between_results is None:
        n_steps_between_results = n_spins

    # shape [n_chains, n_spins]
    init_state = tf.constant(np.random.choice(a=[-1., 1.],
                                              size=(n_spins, )),
                                    dtype=tf.float32)

    generic_kernel = GenericLatticeKernel(n_spins,
                                          energy_fn)

    # Run the chain (with burn-in).
    samples, kernel_results= tfp.mcmc.sample_chain(
        num_results=n_results_per_chain,
        num_burnin_steps=n_burnin_steps,
        num_steps_between_results=n_steps_between_results,
        current_state=init_state,
        parallel_iterations=n_chains,
        kernel=generic_kernel)

    # Accumulate all results along first axis
    samples = tf.reshape(samples, [-1, n_spins])
    kernel_results = tf.reshape(kernel_results, [-1, 1])
    return samples
