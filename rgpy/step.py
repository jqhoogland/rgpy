import os, warnings, math
from functools import reduce
from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import rgpy.samplers
import rgpy.visualize
from rgpy.rbms import *
from rgpy.util import log_timer
from rgpy.standard import BlockRGTransform
#from rgpy.sequence import *

#------------------------------------------------------------------------
#
# RSMI Step
#
#------------------------------------------------------------------------
class LatticeStep(object):
    def __init__(self,
                 J,
                 lattice_width=8,
                 block_size=2,
                 n_samples=1000,
                 name=None):
        """
        Base Class for Block RG methods.

        """

        self.J = J
        self.h = 0
        self.n_samples = n_samples
        self.lattice_width = lattice_width
        self.lattice_shape = [lattice_width, lattice_width]
        self.n_spins = self.lattice_width ** 2
        self.block_size = block_size

        # Make a directory to keep all generated samples/trained models in
        if name is None:
            name = "J={}".format(self.J)

        self.dir_path = name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        # TODO: Generalize to any dimensions / sizes
        self.n_v_spins = self.block_size ** 2
        self.n_e_spins = 4 + 4 * 4
        self.n_ve_spins = self.n_v_spins + self.n_e_spins
        self.n_ve_samples_per_axis = self.lattice_width // self.block_size
        self.n_blocks_per_sample = (self.n_ve_samples_per_axis) ** 2
        self.n_ve_samples = self.n_samples * self.n_blocks_per_sample

        # SAMPLES
        # numpy arrays to fill
        self._x_samples = np.zeros((self.n_samples, self.n_spins), dtype=np.int8)
        self._h_samples = np.zeros((self.n_samples, self.n_blocks_per_sample), dtype=np.int8)
        self._v_samples = np.zeros((self.n_ve_samples, self.n_v_spins), dtype=np.int8)
        self._e_samples = np.zeros((self.n_ve_samples, self.n_e_spins), dtype=np.int8)
        self._ve_samples = np.zeros((self.n_ve_samples, self.n_ve_spins), dtype=np.int8)

        samples_short = ['x', 'h', 'v', 'e', "ve"]
        self.sample_names = ["{}_samples".format(s) for s in samples_short]
        self.sample_save_path_joint = self._wrap_path("samples/samples.npy", "save")
        self.sample_save_paths_separate = [
            self._wrap_path("samples/samples.{}.npy", "save", s) for s in
            samples_short]

        # To track whether we have already created or loaded samples
        self._status = {
            "x_samples": False,
            "h_samples": False,
            "v_samples": False,
            "e_samples": False,
            "ve_samples": False,
        }

        self.expectation_calculator = samplers.Expectation(self.lattice_width)

    def _wrap_path(self, format_str, path_type="save", *format_args, **kwargs):
        path = os.path.join(
            self.dir_path,
            kwargs.get("{}_path".format(path_type), format_str.format(*format_args)))

        dir_path = path[:path.rfind("/")]
        if ("/" in path) and (not os.path.exists(dir_path)):
            os.makedirs(dir_path)

        return path

    def gen_x_samples(self, draw=True, save=True, mcmc="sw"):
        """
        Generates samples of the initial configuration using monte carlo techniques:
            either Metropolis-Hastings, Swendsen-Wang, or Wolff MCMC

        Args:
            draw: whether to draw pictures of generated samples (default=True)
            save: whether to save samples after generating (default=True)
            mcmc (str, one of "mh", "sw", "wf"): which MCMC algorithm to use.
                Metropolis-Hastings, Swendsen-Wang, Wolff, respectively

        """
        self.set_x_samples(
            samplers.ising_generator(
                mcmc=mcmc,
                lattice_width=self.lattice_width,
                J=self.J,
                h=self.h,
                n_results_per_chain=self.n_samples,
                n_chains=1,
                #n_burnin_steps=163,
                #n_steps_between_results=2
            ))

        if draw:
            self.draw_x_samples()

        if save:
            self.save_x_samples()

        return self.get_x_samples()

    def _get_env_units(self, s, block_size, x0, y0, show=False):
        """Returns the environmental units surrounding visible block i, j.
        Does not mask away the hidden units.

        Args:
            s (np.array): a sample
                (assumed to have even number of elements in each direction)
            block_size(int): the size of the blocks
            i (int): horizontal index of the block to be sampled
            j (int): vertical index of the block to be sampled

        Returns:
            np.array: array of blocks top right to bottom left, by order of rows,
                includes P(V) block
        """
        # TODO: extend to generic environment size (currently 3* linear extent)
        blocks = np.ma.array(np.zeros((3 * block_size, 3 * block_size)),
                             fill_value=0, dtype="int")
        n_spins_per_axis = s.shape[0]
        n_blocks_per_axis = n_spins_per_axis // block_size
        """First get all the units in a 3* block_size wide square area surrounding
        the visible block at i, j"""

        # Iterate over the block indices (x,y) relative to visible block (x0,y0)
        for y in [-1, 0, 1]:
            for x in [-1, 0, 1]:
                x_prime = (x0 + x) % n_blocks_per_axis
                y_prime = (y0 + y) % n_blocks_per_axis

                # Correspondign absolute spin coordinates in target config
                i0, j0 = (x + 1)* block_size, (y + 1)* block_size
                i1, j1 = (i0 + 1), (j0 + 1)
                xyth_block = self._get_ijth_block(s, block_size, x_prime, y_prime)
                # TODO: use meshgrid below
                blocks[[i0, i0, i1, i1], [j0, j1, j0, j1]] = xyth_block

        """Then, eliminate all the elements not in the environmental region,
                i.e. the boundary """
        block_idxs = np.arange(3 * block_size)
        block_is, block_js= np.meshgrid(block_idxs, block_idxs)
        is_vertical_env = ((block_is == 0) | (block_is == 5))
        is_horizontal_env = ((block_js == 0) | (block_js == 5))
        blocks.mask= ~(is_vertical_env | is_horizontal_env)

        return blocks[~blocks.mask].data


    def _get_ijth_block(self, sample, block_size, i, j):
        """Returns the block of spins at  target location (i,j)
        When combined with a mask this will allow us to recover the P(V)
        training samples

        Args:
            sample (np.array): array of samples
                (assumed to have even number of elements in each direction)
            block_size(int): the size of the blocks
            i (int): horizontal index of the block to be sampled
            j (int): vertical index of the block to be sampled

        Returns:
            np.array: the block of spins at location i, j
        """
        # get the indices of the top right spin of the block
        n_spins_per_axis = sample.shape[0]
        i0, j0 = block_size * i, block_size * j
        # for now only works with 2x2 block
        # TODO: extend to nxn block
        i1, j1 = (i0 + 1) % (n_spins_per_axis), (j0 + 1) % (n_spins_per_axis)
        ijth_block = sample[[i0, i0, i1, i1], [j0, j1, j0, j1]]

        return ijth_block

    @log_timer("Generating restricted samples of V, E (the visible and environmental blocks)")
    def gen_restricted_samples(self, save=True):
        x_samples = self.get_load_or_create_x_samples()
        x_samples = np.reshape(x_samples, [self.n_samples, *self.lattice_shape])
        ve_samples = np.zeros((self.n_ve_samples, self.n_ve_spins))

        for s_idx, s in enumerate(x_samples):
            for i in range(self.n_ve_samples_per_axis):
                for j in range(self.n_ve_samples_per_axis):
                    ve_sample_idx = (s_idx * self.n_ve_samples_per_axis ** 2
                                     +i * self.n_ve_samples_per_axis + j)
                    ve_samples[ve_sample_idx, :self.n_v_spins] = (
                        self._get_ijth_block(s, self.block_size, i, j)
                    )
                    ve_samples[ve_sample_idx, self.n_v_spins:] = (
                        self._get_env_units(s, self.block_size, i, j)
                    )

        self.set_restricted_samples(ve_samples)

        if save:
            self.save_restricted_samples()

        return self.get_restricted_samples()

    def get_status(self, attr):
        return self._status[attr]

    def set_status(self, attr, new_status=True):
        self._status[attr] = new_status

    def check_status(self, attr, warn=False, ignore=False):
        """Like get_status() but generates an error (warn=False) or warning (warn=True) if status is False."""
        status = self.get_status(attr)
        if not status and not ignore:
            notice = ("""The attribute `{}` has not been changed. Use `run()` to make sure all attributes are properly configured.""".format(attr))
            if attr == "h_samples":
                notice = ("""The hidden samples have not yet been generated. Use an RGTransform() to generate these.""")
            if warn:
                warnings.warn(notice, RuntimeWarning)
            else:
                raise RuntimeError(notice)
        return status

    def get_load_or_create_samples(self):
        return [self.get_load_or_create_x_samples(),
                *self.get_load_or_create_restricted_samples()]

    def save_samples(self):
        # Check whether all necessary samples have been generated
        self.save_x_samples()
        self.save_restricted_samples()
        self.save_h_samples

    def load_samples(self):
        self.load_x_samples()
        self.load_restricted_samples()
        self.load_h_samples

    # X SAMPLES

    def set_x_samples(self, x_samples, new_status=True):
        self._x_samples = np.array(x_samples, dtype=np.int8)
        self.set_status("x_samples", new_status)

    def get_x_samples(self):
        status = self.check_status("x_samples", True)
        return self._x_samples

    def get_load_or_create_x_samples(self):
        status = self.get_status("x_samples")
        if not status:
            print(self._wrap_path("samples/samples.x.npy"))
            if os.path.isfile(self._wrap_path("samples/samples.x.npy")):
                x_samples = self.load_x_samples()
            else:
                x_samples = self.gen_x_samples()
        else:
            x_samples = self._x_samples
        return x_samples

    def save_x_samples(self):
        x_samples = self.get_x_samples()

        save_path = dict(zip(self.sample_names,
                             self.sample_save_paths_separate))['x_samples']

        print('before save', x_samples)
        with open(save_path, 'wb+') as f:
            np.save(f, x_samples)

    def load_x_samples(self, **kwargs):
        load_path = kwargs.get('load_path', self._wrap_path('samples/samples.x.npy'))
        with open(load_path, 'rb') as f:
            print("before ", self._x_samples[0])
            x_samples = np.load(f)
            print("after ", x_samples[0])
            self.set_x_samples(x_samples)

        self.draw_x_samples()
        return x_samples

    def draw_x_samples(self, limit=5000):
        limit = min(self.n_samples, limit)
        shape = (limit // 100, 100)
        visualize.draw_samples(self.get_load_or_create_x_samples(), shape,
                               path=self._wrap_path("samples/images/x_samples.png", "save"))

    # RESTRICTED (i.e. VE) SAMPLES

    def set_restricted_samples(self, ve_samples):
        ve_samples = np.array(ve_samples, dtype=np.int8)
        self._ve_samples = ve_samples
        self._v_samples = ve_samples[:, :4]
        self._e_samples = ve_samples[:, 4:]
        self.set_status("ve_samples", True)

    def get_restricted_samples(self):
        self.check_status('ve_samples', True)
        return [self._v_samples, self._e_samples, self._ve_samples]

    def get_load_or_create_restricted_samples(self):
        status = self.get_status("ve_samples")

        if not status:
            if os.path.isfile(self._wrap_path("samples/samples.ve.npy")):
                self.load_restricted_samples()
            else:
                self.gen_restricted_samples()

        return [self._v_samples, self._e_samples, self._ve_samples]

    def save_restricted_samples(self):
        save_path = self._wrap_path("samples/samples.ve.npy")
        _, _, ve_samples= self.get_restricted_samples()

        with open(save_path, 'wb') as f:
            np.save(f, np.array(ve_samples, dtype=np.int8))

    def load_restricted_samples(self, **kwargs):
        print("Loading restricted sample")
        load_path = kwargs.get('load_path', self._wrap_path('samples/samples.ve.npy'))
        with open(load_path, 'rb') as f:
            ve_samples = np.load(f)

        self.set_restricted_samples(ve_samples)

    # H SAMPLES

    def save_h_samples(self):
        h_samples = self.get_h_samples()

        save_path = dict(zip(self.sample_names,
                             self.sample_save_paths_separate))['h_samples']

        with open(save_path, 'wb+') as f:
            np.save(f, h_samples)

    def load_h_samples(self, **kwargs):
        print("Loading h samaples")
        load_path = kwargs.get('load_path', self._wrap_path('samples/samples.h.npy'))
        with open(load_path, 'rb') as f:
            h_samples = np.load(f)
            self.set_h_samples(h_samples)

    def set_h_samples(self, h_samples):
        self._h_samples = np.array(h_samples, np.int8)
        self.set_status("h_samples", True)

    def get_h_samples(self):
        status = self.check_status('h_samples', True)
        return self._h_samples

    def get_load_or_create_h_samples(self):
        status = self.get_status('h_samples')
        if not status:
            if  os.path.isfile(self._wrap_path("samples/samples.h.npy")):
                self.load_h_samples()
            else:
                raise ValueError("Hidden samples could not be loaded. Generated hidden samples with an RGTransform")

        return self._h_samples

    def draw_h_samples(self, limit=5000):
        limit = min(self.n_samples, limit)
        shape = (limit // 100, 100)
        visualize.draw_samples(self.get_load_or_create_h_samples(), shape,
                               path=self._wrap_path("samples/images/h_samples.png", "save"))


    def set_lattice_width(self, lw):
        self.lattice_width = lw
        self.n_spins = lw ** 2

    def get_expectation(self, fn):
        x_samples = self.get_load_or_create_x_samples()
        print(fn)

        return self.expectation_calculator.get_expectation(fn, x_samples)

    def save(self):
        self.save_x_samples()
        self.save_restricted_samples()
        self.save_h_samples()
