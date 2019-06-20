from typing import List, Callable, NamedTuple, Tuple
import tensorflow as tf
import numpy as np

# Used by RG RBM, TODO: CONSOLIDATE
def expectation(samples: List[List[tf.Tensor]],
                fn: Callable[[List[tf.Tensor]], tf.Tensor]) -> tf.Tensor:
    """ tfp.monte_carlo.expectation didn't have the right functionality--
    I wanted to be able to calculate multiple expectation values concurrently.

    Args:
        samples
    """
    return fn(samples)
#return tf.reduce_mean(fn(samples), axis=0)

def expectations(samples: List[List[tf.Tensor]],
                 fns: List[Callable[[List[tf.Tensor]], tf.Tensor]]) -> List[tf.Tensor]:
    return [expectation(samples, fn) for fn in fns]


# USED BY RSMI
class Expectation(object):
    def __init__(self, lattice_width):
        self.set_lattice_width(lattice_width)
        self.samples = tf.placeholder(tf.float32)

    def set_lattice_width(self, lattice_width):
        self.lattice_width = lattice_width
        self.n_spins = lattice_width ** 2

    def get_susceptibility(self, samples):
        """ Returns the magnetic susceptibility:
        chi = <s_0 s_j> - < s_0>^2

        Args:
            samples (tf.Tensor of shape [batch_size, lattice_width**2])
        """
        # We compute chi per sample (so we can get both mean and std)
        # Take outer product to get array of shape
        # [batch_size, lattice_width**2, lattice_width**2]
        # collapse last two axes, then average over this axis.
        print(samples[0], (np.expand_dims(samples, 2) * np.expand_dims(samples, 1))[0])
        M_squared = float(self.get_magnetization(samples)['mean']) ** 2
        sisj = np.mean(np.reshape(np.expand_dims(samples, 2) * np.expand_dims(samples, 1),
                       [samples.shape[0], -1]), axis=1)
        chi = sisj - M_squared
        return {'mean': str(np.mean(chi)), 'std': str(np.std(chi))}

    def get_magnetization(self, samples, absolute_value=tf.constant(True)):
        """ Returns the magnetization of
        absolute_value is True, has no effect
        """
        M = np.abs(np.mean(samples, axis=1))
        return {'mean': str(np.mean(M)), 'std': str(np.std(M))}

    def get_nn_correlation(self, samples, bdry_conds='helical'):
        """ Returns the nearest-neighbors correlation
        We have to divide by two because we some over 2 direction within each average
        """
        if bdry_conds == "helical":
            nn_correlation = lambda x: tf.reduce_mean(
                (x * tf.roll(x, 1, axis=1) + x * tf.roll(x, self.lattice_width, axis=1)),
                axis=1) / 2
        elif bdry_conds == "toroidal":
            def nn_correlation(x):
                x_square = tf.reshape(x, [-1, self.lattice_width, self.lattice_width])
                return tf.reduce_mean(tf.reshape(
                    (x_square * tf.roll(x_square, 1, 1) + x_square * tf.roll(x_square, 1, 2)),
                    [-1, self.n_spins]), axis=1) / 2
        else:
            raise NotImplementedError("Boundary conditions {} not implemented.".format(bdry_conds))
        return self.get_expectation(nn_correlation, samples)


    def get_nnn_correlation(self, samples, bdry_conds='helical'):
        """ Returns the next-nearest neighbors correlation
        """
        if bdry_conds == "helical":
            nnn_correlation = lambda x: tf.reduce_mean(
                (x * tf.roll(x, self.lattice_width + 1, axis=1) + x * tf.roll(x, self.lattice_width - 1, axis=1)),
                axis=1) / 2
        elif bdry_conds == "toroidal":
            def nnn_correlation(x):
                x_square = tf.reshape(x, [-1, self.lattice_width, self.lattice_width])
                return tf.reduce_mean(tf.reshape(
                    (x_square * tf.roll(tf.roll(x_square, 1, 1), 1, 2) +
                     x_square * tf.roll(tf.roll(x_square, 1, 1)), -1, 2),
                    [-1, self.n_spins]), axis=1) / 2
        else:
            raise NotImplementedError("Boundary conditions {} not implemented.".format(bdry_conds))
        return self.get_expectation(nnn_correlation, samples)

    def get_expectation(self, fn, samples):
        """ Returns the expectation of `fn` over x_samples.

        Args:
            fn: A tf function with placeholder `samples`;
                Should take input shape [?, self.n_spins]
                and return output shape [?, 1] or [?]
        """
        samples = np.array(samples).reshape([-1, self.n_spins])
        if isinstance(fn, str):
            if fn == 'magnetization':
                return self.get_magnetization(samples)
            elif fn == 'nn_correlation':
                return self.get_nn_correlation(samples)
            elif fn == 'nnn_correlation':
                return self.get_nnn_correlation(samples)
            elif fn == 'susceptibility':
                return self.get_susceptibility(samples)

        mean, std = tf.nn.moments(fn(self.samples), axes=[0])
        mean_, std_ = tf.Session().run([mean, std],feed_dict={self.samples: samples})
        return {'mean': str(mean_), 'std': str(std_)}
