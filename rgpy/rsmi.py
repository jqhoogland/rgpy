import os, warnings
from functools import reduce
from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import rsmi_renormalization.samplers
import rsmi_renormalization.visualize
from rsmi_renormalization.rbms import *
from rsmi_renormalization.util import log_timer
from rsmi_renormalization.standard import BlockRGTransform

default_psi_rbm = {
    "n_hidden":2,
    "learning_rate": 0.001,
    "lmbda":0.01,
    "k":1,
    "use_tqdm":True,
    "binary": [-1, 1]
}

default_psi_rbm_fit = {
    "n_epochs": 10,
    "batch_size": 1000,
}

default_theta_rbm = {
    "n_hidden":10,
    "learning_rate": 0.001,
    "lmbda":0.01,
    "k":1,
    "use_tqdm":True,
    "binary": [-1, 1]
}


default_theta_rbm_fit = {
    "n_epochs": 10,
    "batch_size": 1000,
}

default_lambda_rbm = {
    "n_hidden":1,
    "lmbda":0.01,
    "use_tqdm":True,
    "binary": [-1, 1]

}

default_lambda_rbm_fit = {
    "batch_size": 1000,
}

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

    def gen_h_samples(self, activate=True, draw=True, save=True):
        raise NotImplementedError

    def get_status(self, attr):
        return self._status[attr]

    def set_status(self, attr, new_status=True):
        self._status[attr] = new_status

    def check_status(self, attr, warn=False, ignore=False):
        """Like get_status() but generates an error (warn=False) or warning (warn=True) if status is False."""
        status = self.get_status(attr)
        if not status and not ignore:
            notice = ("""The attribute `{}` has not been changed. Use `run()` to make sure all attributes are properly configured.""".format(attr))
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
                self.gen_h_samples()

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

class MajorityRuleStep(LatticeStep):
    def __init__(self,
                 J,
                 lattice_width=8,
                 block_size=2,
                 n_samples=1000,
                 name=None):
        """
        """
        LatticeStep.__init__(self, J, lattice_width, block_size, n_samples, name)
        self.block_rg = BlockRGTransform(block_size)

    @log_timer("Generating samples of H (the coarse grained-system)")
    def gen_h_samples(self, activate=True, draw=True, save=True):
        [v_samples, _, _]=self.get_load_or_create_restricted_samples()
        h_samples_raw=self.block_rg.transform_nonbinary(v_samples)

        self.set_h_samples(h_samples_raw.reshape(self.n_samples, self.n_blocks_per_sample))

        if draw:
            visualize.draw_samples(self._h_samples, (1, self.n_samples),
                                   path=self._wrap_path("samples/images/h_samples.png", "save"))

        if save:
            self.save_h_samples()

        return self.get_h_samples()

    @log_timer("Running one RSMI algorithm step.")
    def run(self, overwrite=False, gen_x_samples=True, draw_sample=True):
        # This will recursively check for all dependencies through
        # get_load_or_create and get_load_or_train
        self.get_load_or_create_samples()
        self.get_load_or_create_h_samples()
        self.save()

        if draw_sample:
            self.draw_x_samples()
            self.draw_h_samples()

        return 0


# ------------------------------------------------------------------------------
# RSMI STEP
# ------------------------------------------------------------------------------

# TODO save_path/load_path= None replace with decorator
class RSMIStep(LatticeStep):
    def __init__(self,
                 J,
                 lattice_width=8,
                 block_size=2,
                 n_samples=1000,
                 name=None,
                 settings_psi_rbm=default_psi_rbm,
                 settings_psi_rbm_fit=default_psi_rbm_fit,
                 settings_theta_rbm=default_theta_rbm,
                 settings_theta_rbm_fit=default_theta_rbm_fit,
                 settings_lambda_rbm=default_lambda_rbm,
                 settings_lambda_rbm_fit=default_lambda_rbm_fit):

        # TODO: J may not be known beforehand (if we provide the step with x_samples).
        #       Allow J to take non-specified value

        LatticeStep.__init__(self, J, lattice_width, block_size, n_samples, name)


        # RBMS
        self.psi_rbm = BBRBM(
            n_visible=self.n_v_spins,
            **settings_psi_rbm
        )

        self.theta_rbm = BBRBM(
            n_visible=self.n_ve_spins,
            **settings_theta_rbm
        )
        # TODO: load these as well?
        self._psi_energy_fn=lambda v: self.psi_rbm.get_visible_energy(v)


        self._theta_energy_fn = lambda v, e: (
            self.theta_rbm.get_visible_energy(
                tf.concat([tf.reshape(v, [2, self.n_v_spins]),
                           e], #tf.reshape(e [-1, self.n_e_spins])],
                          axis=1)))

        self.lambda_rbm=RGRBM(
            self._psi_energy_fn,
            self._theta_energy_fn,
            n_visible=self.n_v_spins,
            n_environment=self.n_e_spins,
            **settings_lambda_rbm
        )

        # For use by self.save_rbms(), self.load_rbms()
        self.rbm_names = ["lambda", "psi", "theta"]
        self.rbm_save_paths = [
            self._wrap_path("rbms/{0}/{0}.ckpt", "save", name) for name in self.rbm_names]

        self.rbm_training_settings = {'lambda': settings_lambda_rbm_fit,
                                      'psi': settings_psi_rbm_fit,
                                      'theta':settings_theta_rbm_fit}

        # For use by self.save_samples(), self.load_samples()

        # Tracks information about whether relevant models have been trained and
        # whether samples have been generated/processed

        self._status.update({
            "lambda": False,
            "psi": False,
            "theta": False,
        })

    @log_timer("Generating samples of H (the coarse grained-system)")
    def gen_h_samples(self, activate=True, draw=True, save=True):
        [v_samples, _, _]=self.get_load_or_create_restricted_samples()
        lambda_rbm=self.get_load_or_train_rbm("lambda")

        h_samples_raw=lambda_rbm.transform_nonbinary(v_samples, activate)

        self.set_h_samples(h_samples_raw.reshape(self.n_samples, self.n_blocks_per_sample))

        if draw:
            visualize.draw_samples(self._h_samples, (1, self.n_samples),
                                   path=self._wrap_path("samples/images/h_samples.png", "save"))

        if save:
            self.save_h_samples()

        return self.get_h_samples()

    def train_rbm(self, name, save=True):
        rbm = self.get_rbm(name, ignore=True)
        @log_timer("Training the {}-RBM.".format(name))
        def _train_rbm():
            if name == "lambda":
                theta_rbm = self.get_load_or_train_rbm('theta')
                psi_rbm = self.get_load_or_train_rbm('psi')

            print_filters_dir = self._wrap_path("rbms/{}/filters/", "save", name)

            rbm.fit(self.get_rbm_training_data(name),
                    print_filters_dir=print_filters_dir,
                    **self.get_rbm_training_settings(name))

            self.set_status(name, True)
            self.set_rbm(name, rbm)
            self.update_lambda_energy_fn(name)

            return rbm

        rbm = _train_rbm()

        if save:
            self.save_rbm(name)

        return rbm

    def update_lambda_psi_energy_fn(self):
        self.lambda_rbm.energy_psi_of_v=self._psi_energy_fn

    def update_lambda_theta_energy_fn(self):
        self.lambda_rbm.energy_theta_of_ve=self._theta_energy_fn

    def update_lambda_energy_fn(self, name):
        if name == "psi":
            self.update_lambda_psi_energy_fn()
        elif name == "theta":
            self.update_lambda_theta_energy_fn()

    @log_timer("Running one RSMI algorithm step.")
    def run(self, overwrite=False, gen_x_samples=True, draw_sample=True):
        if overwrite:
            # Generate samples of the entire configuration
            if gen_x_samples:
                self.gen_x_samples()
                self.save_x_samples()

            # Process samples of full configuration into restricted samples
            self.gen_restricted_samples()
            self.save_restricted_samples()

            # Train the RBMs
            for rbm_name in ['psi', 'theta', 'lambda']:
                self.train_rbm(rbm_name)
                self.save_rbm(rbm_name)

            # Generate samples of the next layer with the trained Lambda-RBM
            self.gen_h_samples(activate=True)
            self.save_h_samples()
        else:
            # This will recursively check for all dependencies through
            # get_load_or_create and get_load_or_train
            self.get_load_or_create_samples()
            self.get_load_or_train_rbms()
            self.get_load_or_create_h_samples()
            self.save()

        if draw_sample:
            self.draw_x_samples()
            self.draw_h_samples()

        return self.lambda_rbm.mutual_info


    def get_rbm(self, name, ignore=False):
        self.check_status(name, warn=True, ignore=False)
        return dict(zip(self.rbm_names, self.get_rbms()))[name]

    def get_load_or_train_rbm(self, name):
        status = self.get_status(name)
        rbm = self.get_rbm(name, ignore=True)
        if not status:
            if os.path.isfile(self._wrap_path("rbms/{0}/{0}.ckpt.meta", "save", name)):
                self.load_rbm(name)
            else:
                rbm = self.train_rbm(name)
        return rbm

    def get_load_or_train_rbms(self):
        return [get_load_or_train_rbm(rbm) for rbm in self.rbm_names]

    def get_rbms(self):
        return [self.lambda_rbm, self.psi_rbm, self.theta_rbm]

    def set_rbm(self, name, rbm):
        rbm_dict = dict(zip(self.rbm_names, self.get_rbms()))
        rbm_dict[name] = rbm
        self.set_rbms(*rbm_dict.values())

    def set_rbms(self, lambda_rbm, psi_rbm, theta_rbm):
        self.lambda_rbm, self.psi_rbm, self.theta_rbm=lambda_rbm, psi_rbm, theta_rbm

    def save_rbm(self, rbm_name):
        save_path = dict(zip(self.rbm_names, self.rbm_save_paths))[rbm_name]
        rbm = self.get_rbm(rbm_name)
        rbm.save_weights(save_path, rbm_name)
        self.set_rbm(rbm_name, rbm)

    def load_rbm(self, name, rbm_dir="rbms"):
        load_path = self._wrap_path("{0}/{1}/{1}.ckpt.meta", "save", rbm_dir, name)

        rbm = self.get_rbm(name)
        rbm.load_weights(self._wrap_path("{0}/{1}/{1}.ckpt", "load", rbm_dir, name), name)

        return rbm

    def save_rbms(self):
        for rbm_name in self.rbm_names:
            self.check_status(rbm_name, True)

        rbms = self.get_rbms()
        for rbm, rbm_name, rbm_save_path in zip(rbms, self.rbm_names, self.rbm_save_paths):
            rbm.save_weights(rbm_save_path, rbm_name)

    def load_rbms(self, rbm_dir="rbms"):
        load_paths = [
            self._wrap_path("{0}/{1}/{1}.ckpt.meta", "save", rbm_dir, name)
            for name in self.rbm_names]

        rbms = self.get_rbms()
        for rbm, rbm_name, rbm_load_path in zip(rbms, self.rbm_names, load_paths):
            rbm.load_weights(rbm_load_path, rbm_name)

        return self.rbms

    def get_rbm_training_data(self, name):
        # TODO: separate training and test data
        [v, _, ve] = self.get_load_or_create_restricted_samples()
        return {'lambda': ve,
                'psi': v,
                'theta': ve}[name]

    def get_rbm_training_settings(self, name):
        return self.rbm_training_settings[name]

    def save(self):
        self.save_x_samples()
        self.save_restricted_samples()
        self.save_h_samples()
        self.save_rbms()


#------------------------------------------------------------------------
#
# RSMI Sequence
#
#------------------------------------------------------------------------

# TODO: use keras.Sequential-like functionality
class RSMISequence(object):
    def __init__(self,
                 J,
                 lattice_width=8,
                 n_samples=1000,
                 block_size=2,
                 n_steps=3,
                 name=None,
                 procedure="rsmi",
                 **kwargs):
        """
        Args:
            J (float): thermal parameter of Ising model
            lattice_width
            n_samples
            block_size
            n_steps
            name
            procedure (str, one of "rsmi", "majority-rule"): the kind of RG procedure to perform
        """

        # Make a directory to keep all generated samples/trained models in
        self.J = J
        self.n_samples = n_samples
        self.init_lattice_width = lattice_width
        self.init_lattice_shape = [self.init_lattice_width,
                                   self.init_lattice_width]
        self.init_n_spins = self.init_lattice_width ** 2
        self.block_size = block_size
        self.n_steps = n_steps

        self.procedure = procedure

        self.metadata = {i: {} for i in range(n_steps)}

        if name is None:
            name = "J={}".format(self.J)

        self.dir_path = name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        def prepare_step(i):
            sub_dir_path = os.path.join(self.dir_path, str(i))
            _lattice_width = int(self.init_lattice_width // (2 ** i))
            _n_spins = int(self.init_n_spins // (2 ** i))

            # TODO: allow non-default settings for the rbms
            if self.procedure == "rsmi":
                step = RSMIStep(J,
                                _lattice_width,
                                self.block_size,
                                self.n_samples,
                                name=sub_dir_path,
                                **kwargs)
            elif self.procedure == "majority-rule":
                step = MajorityRuleStep(J,
                                        _lattice_width,
                                        self.block_size,
                                        self.n_samples,
                                        name=sub_dir_path,
                                        **kwargs)
            else:
                raise ValueError("Procedure {} not recognized or not yet implemented.".format(self.procedure))

            return step

        self.rsmi_steps = [prepare_step(i) for i in range(n_steps)]

    def _wrap_path(self, path):
        return os.path.abspath(os.path.join(self.dir_path, path))

    @log_timer("GENERATING SAMPLES FOR STEP 0")
    def gen_samples(self, overwrite=False, mcmc="sw"):
        self.rsmi_steps[0].gen_x_samples(draw=True, save=True, mcmc=mcmc)

    def load_samples(self):
        self.rsmi_steps[0].load_x_samples(load_path=self._wrap_path("0/samples/samples.x.npy"))
        self.rsmi_steps[0].load_restricted_samples()

    @log_timer("RUNNING STEP 0")
    def run_first_step(self, overwrite=False, gen_samples=False):
        self.rsmi_steps[0].run(overwrite=overwrite, gen_x_samples=gen_samples)

    def run(self, overwrite=False, gen_init_samples=False, draw=True):
        self.run_first_step(overwrite, gen_init_samples)

        # TODO: something nicer with reduce?
        for i, (curr_step, prev_step) in enumerate(zip(self.rsmi_steps[1:],
                                                       self.rsmi_steps[:-1])):
            @log_timer("RUNNING STEP {}".format(i + 1))
            def run_step_i():
                if not os.path.isfile(self._wrap_path("{}/samples/samples.x.npy".format(i + 1))):
                    os.symlink(self._wrap_path("{}/samples/samples.h.npy".format(i)),
                               self._wrap_path("{}/samples/samples.x.npy".format(i + 1)))
                return curr_step.run(overwrite=overwrite, gen_x_samples=False)

            mutual_info = run_step_i()
            self.metadata[i]["mutual_info"] = str(mutual_info)

        if draw:
            self.draw_evolution()

    def draw_evolution(self):

        get_image_path = lambda i, kind: self._wrap_path("{}/samples/images/{}_samples.png".format(i, kind))

        get_hidden_path = lambda i: get_image_path(i, 'h')

        im_0 = Image.open(get_image_path(0, 'x'))
        im_0_x, im_0_y= im_0.size

        print(im_0_x, im_0_y)

        new_x, new_y = im_0_x, im_0_y * (self.n_steps + 1)
        new_im = Image.new('RGB', (new_x, new_y))
        new_im.paste(im_0)

        for i in range(self.n_steps):
            im_i = Image.open(get_hidden_path(i)).resize((im_0_x, im_0_y))
            new_im.paste(im_i, (0, (im_0_y * (i + 1))))

        new_im.save(os.path.join(self.dir_path, 'evolution.png'))

    def get_expectation(self, fn, **kwargs):
        """Computes the expectation of fn over the samples for each step
        """
        steps = kwargs.get("steps", range(self.n_steps))

        expectations = [rsmi_step.get_expectation(fn) for rsmi_step in self.rsmi_steps]

        for i in steps:
            _metadata = self.metadata[i]
            _metadata[fn] = str(expectations[i])
            self.metadata[i] = _metadata

        return expectations

#------------------------------------------------------------------------
#
# RSMI
#
#------------------------------------------------------------------------

class RSMI(object):
    def __init__(self,
                 Js,
                 lattice_width=8,
                 n_samples=1000,
                 block_size=2,
                 n_steps=3,
                 name=None,
                 **kwargs):
        """
        """

        # Make a directory to keep all generated samples/trained models in
        self.Js = Js
        self.n_samples = n_samples
        self.init_lattice_width = lattice_width
        self.init_lattice_shape = [self.init_lattice_width,
                                   self.init_lattice_width]
        self.init_n_spins = self.init_lattice_width ** 2
        self.block_size = block_size
        self.n_steps = n_steps

        self.metadata = dict([str(J), {}] for J in self.Js)
        print(self.metadata)

        if name is None:
            name = "rsmi_run"

        self.name = name

        self.dir_path = name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        def prepare_sequence(i):
            J = self.Js[i]
            sub_dir_path = os.path.join(self.dir_path, "J={}".format(J))
            # TODO: allow non-default settings for the rbms
            return RSMISequence(J=J,
                                lattice_width=self.init_lattice_width,
                                n_samples=self.n_samples,
                                block_size=self.block_size,
                                n_steps=self.n_steps,
                                name=sub_dir_path,
                                **kwargs)

        self.rsmi_sequences = [prepare_sequence(i) for i in range(len(self.Js))]

    def run_sequence(self, i, overwrite=False, gen_init_samples=False, expectation_fns=[]):
        self.rsmi_sequences[i].run(overwrite, gen_init_samples)

        for fn in expectation_fns:
            self.rsmi_sequences[i].get_expectation(fn)

        self.update_metadata(i)

    def run(self, overwrite=False, gen_init_samples=True, expectation_fns=[]):
        for i in range(len(self.Js)):
            @log_timer("RUNNING SEQUENCE {} WITH J={}".format(i, self.Js[i]))
            def _run_sequence():
                self.run_sequence(i, overwrite, gen_init_samples, expectation_fns)

            _run_sequence()

    def gen_samples(self, overwrite=False, expectation_fns=[], mcmc="sw"):
        for i in range(len(self.Js)):
            @log_timer("GENERATING SAMPLES FOR SEQUENCE {} WITH J={}".format(i, self.Js[i]))
            def _run_sequence():
                self.rsmi_sequences[i].gen_samples(overwrite=overwrite, mcmc=mcmc)

                for fn in expectation_fns:
                    self.rsmi_sequences[i].get_expectation(fn, steps=[0])

                self.update_metadata(i)

            _run_sequence()

    def load_samples(self, expectation_fns=[]):
        for i, sequence in enumerate(self.rsmi_sequences):
            sequence.load_samples()

            for fn in expectation_fns:
                sequence.get_expectation(fn)

            self.update_metadata(i)

    def get_expectations(self, expectation_fns=[], steps=[0]):
        for i, rsmi_sequence in enumerate(self.rsmi_sequences):
            for fn in expectation_fns:
                rsmi_sequence.get_expectation(fn, steps=steps)
                self.update_metadata(i)


    def update_metadata(self, i):
        self.metadata[str(self.Js[i])] = { **self.metadata[str(self.Js[i])], **self.rsmi_sequences[i].metadata }
        self.write_metadata()

    def write_metadata(self, **kwargs):
        import json

        save_path = kwargs.get('save_path', 'metadata.json')
        with open("{}/{}".format(self.name, save_path), 'w+') as f:
            json.dump(self.metadata, f)
