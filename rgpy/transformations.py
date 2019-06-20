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

default_psi_rbm = {
    "n_hidden":2,
    "learning_rate": 0.025,
    "lmbda":0.01,
    "k":1,
    "use_tqdm":True,
    "binary": [-1, 1]
}

default_psi_rbm_fit = {
    "n_epochs": 100,
    "batch_size": 25,
}

default_theta_rbm = {
    "n_hidden":10,
    "learning_rate": 0.025,
    "lmbda":0.01,
    "k":1,
    "use_tqdm":True,
    "binary": [-1, 1]
}


default_theta_rbm_fit = {
    "n_epochs": 100,
    "batch_size": 25,
}

default_lambda_rbm = {
    "n_hidden":1,
    "lmbda":0.01,
    "use_tqdm":True,
    "binary": [-1, 1]
}

default_lambda_rbm_fit = {
    "n_epochs": 100,
    "batch_size": 800,
}

class RGTransform(object):
    """
    Perform RG transformations on Lattice Steps.
    Separates boilerplate for manipulating and storing files of samples
    and the logic of transforming these samples.

    Children of this class will include particular implementations like:
        - Block-spin renormalization
        - Real-space Mutual Information Maximization
    """

    def __init__(self,
                 lattice_width=8,
                 block_size=2,
                 n_samples=1000):
        "docstring"
        self.n_samples = n_samples
        self.lattice_width = lattice_width
        self.lattice_shape = [lattice_width, lattice_width]
        self.n_spins = self.lattice_width ** 2
        self.block_size = block_size

        # TODO: Generalize to any dimensions / sizes
        self.n_v_spins = self.block_size ** 2
        self.n_e_spins = 4 + 4 * 4
        self.n_ve_spins = self.n_v_spins + self.n_e_spins
        self.n_ve_samples_per_axis = self.lattice_width // self.block_size
        self.n_blocks_per_sample = (self.n_ve_samples_per_axis) ** 2
        self.n_ve_samples = self.n_samples * self.n_blocks_per_sample

    def _transform(self, v_samples):
        raise NotImplementedError

    @log_timer("Generating samples of H (the coarse grained-system)")
    def transform(self, v_samples):
        h_samples_raw=self._transform(v_samples)
        return h_samples_raw.reshape([self.n_samples, -1])

    def _run_step(self, step):
        """ By default assumes there is no need to load/train procedure
        """
        [v, _, _] = step.get_load_or_create_restricted_samples()
        step.set_h_samples(self.transform(v))
        return step

    def run_step(self, step, save=True, draw=True):
        """
        Wrapper for _run_step which also allows for saving and drawing

        Args:
            step (LatticeStep): the step to perform RG on
        """
        step = self._run_step(step)

        if save:
            step.save_h_samples()

        if draw:
            step.draw_h_samples()

        return step.get_h_samples()

class MajorityRuleTransform(RGTransform):
    def __init__(self,
                 lattice_width=8,
                 block_size=2,
                 n_samples=1000):
        """
        """
        RGTransform.__init__(self, lattice_width, block_size, n_samples)
        self.block_rg = BlockRGTransform(block_size)

    def _transform(self, v_samples):
        return self.block_rg.transform_nonbinary(v_samples)

# ------------------------------------------------------------------------------
# RSMI STEP
# ------------------------------------------------------------------------------

# TODO save_path/load_path= None replace with decorator
class RSMITransform(RGTransform):
    def __init__(self,
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

        RGTransform.__init__(self, lattice_width, block_size, n_samples)

        # Make a directory to keep all generated samples/trained models in
        if name is None:
            name = "rbms".format()

        self.dir_path = name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

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
        self.rbm_save_paths = {
            "theta":self._wrap_path("theta/theta.ckpt"),
            "psi": self._wrap_path("psi/psi.ckpt"),
            "lambda":self._wrap_path("lambda/lambda.ckpt")
        }

        self.rbm_training_settings = {'lambda': settings_lambda_rbm_fit,
                                      'psi': settings_psi_rbm_fit,
                                      'theta':settings_theta_rbm_fit}

        # For use by self.save_samples(), self.load_samples()

        # Tracks information about whether relevant models have been trained and
        # whether samples have been generated/processed

        self._status = {
            "lambda": False,
            "psi": False,
            "theta": False,
        }

    def _transform(self, v_samples):
        """ Assumes lambda rbm (and thus theta-rbms) has already been trained.
        v_samples [n_samples, 4] -> h_samples [n_samples, 1].
        Reorganizing in configurations is the task of self.transform
        """
        return self.lambda_rbm.transform_nonbinary(v_samples, True)

    def _run_step(self, step):
        """
        Args:
            step (LatticeStep): the step to perform RG on
        """
        [v, _, ve] = step.get_load_or_create_restricted_samples()
        self.get_load_or_train_lambda_rbm(v, ve)
        step.set_h_samples(self.transform(v))
        return step

    #
    # HELPER METHODS TO CHECK STATUS: WHETHER RBMS HAVE BEEN LOADED
    #

    def get_status(self, attr):
        """ Helper method for keeping track of whether RBMs have been trained
        """
        return self._status[attr]

    def set_status(self, attr, new_status=True):
        self._status[attr] = new_status

    def check_status(self, attr, warn=False, ignore=False):
        """Like get_status() but generates an error (warn=False) or warning
        (warn=True) if status is False."""
        status = self.get_status(attr)
        if not status and not ignore:
            notice = (("The attribute `{}` has not been changed." +
                       "Use `run()` to make sure all attributes are properly configured.")
                      .format(attr))
            if warn:
                warnings.warn(notice, RuntimeWarning)
            else:
                raise RuntimeError(notice)
        return status

    #
    # UPDATE RBM WRAPPER FUNCTIONS
    #

    def _update_theta(self, save=True):
        """ To be called whenever theta is updated by either loading or training theta.
        """
        self.set_status('theta')
        self.lambda_rbm.energy_theta_of_ve=self._theta_energy_fn

        if save:
            self.save_theta_rbm()

        return self.theta_rbm

    def _update_psi(self, save=True):
        self.set_status('psi')
        self.lambda_rbm.energy_psi_of_v=self._psi_energy_fn

        if save:
            self.save_psi_rbm()

        return self.psi_rbm

    def _update_lambda(self, save=True):
        self.set_status('lambda')

        if save:
            self.save_lambda_rbm()

        return self.lambda_rbm

    def _train_rbm(self, rbm, name, data, **settings):
        print_filters_dir = self._wrap_path("{}/filters/".format(name))
        rbm.fit(data,
                print_filters_dir=print_filters_dir,
                **settings)
        return rbm
    #
    # TRAIN METHODS
    #

    def train_theta_rbm(self, ve, **kwargs):
        settings = default_theta_rbm_fit
        settings.update(kwargs)

        self._train_rbm(self.theta_rbm, 'theta', ve)
        return self._update_theta()

    def train_psi_rbm(self, v, **kwargs):
        settings = default_psi_rbm_fit
        settings.update(kwargs)

        self._train_rbm(self.psi_rbm, 'psi', v)
        return self._update_psi()

    def train_lambda_rbm(self, v, ve, **kwargs):
        settings = default_lambda_rbm_fit
        settings.update(kwargs)

        theta_rbm = self.get_load_or_train_theta_rbm(ve)
        psi_rbm = self.get_load_or_train_psi_rbm(v)
        self._train_rbm(self.lambda_rbm, 'lambda', ve, **kwargs)
        return self._update_lambda()

    #
    # LOAD METHODS
    #

    def _load_path(self, name):
        return self.rbm_save_paths[name]

    def load_theta_rbm(self):
        self.theta_rbm.load_weights(self._load_path('theta'), 'theta')
        return self._update_theta()

    def load_psi_rbm(self):
        self.psi_rbm.load_weights(self._load_path('psi'), 'psi')
        return self._update_psi()

    def load_lambda_rbm(self):
        self.lambda_rbm.load_weights(self._load_path('lambda'), 'lambda')
        return self._update_lambda()

    #
    # GET METHODS
    #

    def get_theta_rbm(self, ignore=False):
        self.check_status('theta', warn=True, ignore=False)
        return self.theta_rbm

    def get_psi_rbm(self, ignore=False):
        self.check_status('psi', warn=True, ignore=False)
        return self.psi_rbm

    def get_lambda_rbm(self, ignore=False):
        self.check_status('lambda', warn=True, ignore=False)
        return self.lambda_rbm

    def get_rbm_training_settings(self, name):
        return self.rbm_training_settings[name]

    #
    # GET LOAD OR TRAIN METHODS
    #

    def get_load_or_train_theta_rbm(self, ve_samples):
        rbm = self.get_theta_rbm(ignore=True)
        if not self.get_status('theta'):
            try:
                rbm = self.load_theta_rbm()
            except:
                rbm = self.train_theta_rbm(ve_samples)

        return rbm

    def get_load_or_train_psi_rbm(self, v_samples):
        rbm = self.get_psi_rbm(ignore=True)
        if not self.get_status('psi'):
            try:
                rbm = self.load_psi_rbm()
            except:
                rbm = self.train_psi_rbm(v_samples)

        return rbm

    def get_load_or_train_lambda_rbm(self, v_samples, ve_samples):
        """
        Args:
            v_samples: possibly needed if psi rbm hasn't been trained
            ve_samples:
        """
        rbm = self.get_lambda_rbm(ignore=True)
        if not self.get_status('lambda'):
            try:
                rbm = self.load_lambda_rbm()
            except:
                rbm = self.train_lambda_rbm(v_samples, ve_samples)

        return rbm

    #
    # SAVE METHODS
    #

    def save(self):
        for rbm_name in self.rbm_names:
            self.check_status(rbm_name, True)

        rbms = [self.psi_rbm, self.theta_rbm, self.psi_rbm]
        for rbm, rbm_name, rbm_save_path in zip(rbms, self.rbm_names, self.rbm_save_paths):
            rbm.save_weights(rbm_save_path, rbm_name)

    def save_theta_rbm(self):
        self.theta_rbm.save_weights(self.rbm_save_paths['theta'], 'theta')

    def save_psi_rbm(self):
        self.psi_rbm.save_weights(self.rbm_save_paths['psi'], 'psi')

    def save_lambda_rbm(self):
        self.lambda_rbm.save_weights(self.rbm_save_paths['lambda'], 'lambda')

    def _wrap_path(self, path):
        path = os.path.join(
            self.dir_path,
           path)

        # Make all necessary paths
        dir_path = path[:path.rfind("/")]
        if ("/" in path) and (not os.path.exists(dir_path)):
            os.makedirs(dir_path)

        return os.path.abspath(path)
