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
from rgpy.step import *
from rgpy.transformations import *

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

        self.metadata = {i: {} for i in range(n_steps)}

        """ Initialize directory to store generated samples
        """
        if name is None:
            name = "J={}".format(self.J)

        self.dir_path = name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        def prepare_step(i):
            sub_dir_path = os.path.join(self.dir_path, str(i))
            _lattice_width = int(self.init_lattice_width // (2 ** i))
            _n_spins = int(self.init_n_spins // (2 ** i))

            step = LatticeStep(J,
                                _lattice_width,
                                self.block_size,
                                self.n_samples,
                                name=sub_dir_path,
                                **kwargs)

            return step

        self.rsmi_steps = [prepare_step(i) for i in range(n_steps)]

    def _wrap_path(self, path):
        return os.path.abspath(os.path.join(self.dir_path, path))

    @log_timer("GENERATING SAMPLES FOR STEP 0")
    def gen_samples(self, overwrite=False, mcmc="sw"):
        self.rsmi_steps[0].gen_x_samples(draw=True, save=True, mcmc=mcmc)

    def load_samples(self):
        """ Loads only the first set of samples
        """
        self.rsmi_steps[0].load_x_samples(load_path=self._wrap_path("0/samples/samples.x.npy"))
        self.rsmi_steps[0].load_restricted_samples()

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
            _metadata[fn] = expectations[i]
            self.metadata[i] = _metadata

        return expectations

    def get_step(self, i):
        return self.rsmi_steps[i]

    def link_steps(self, i):
        src_samples_path = self._wrap_path('{}/samples/samples.h.npy'.format(i))
        dst_samples_path = self._wrap_path('{}/samples/samples.x.npy'.format(i + 1))
        if not os.path.isfile(dst_samples_path) and i < self.n_steps - 1:
            os.symlink(src_samples_path, dst_samples_path)

#------------------------------------------------------------------------
#
# RSMI
#
#------------------------------------------------------------------------

class RSMI(object):
    def __init__(self, Js,
                 lattice_width=8,
                 n_samples=1000,
                 block_size=2,
                 n_steps=3,
                 lambda_rbm=False,
                 name=None,
                 procedure='rsmi',
                 **kwargs):
        """
        """

        self.Js = Js
        self.n_samples = n_samples
        self.init_lattice_width = lattice_width
        self.init_lattice_shape = [self.init_lattice_width,
                                   self.init_lattice_width]
        self.init_n_spins = self.init_lattice_width ** 2
        self.block_size = block_size
        self.n_steps = n_steps

        # stores information about expectation values
        self.metadata = dict([str(J), {}] for J in self.Js)

        # Name specifies for storing the samples and trained RBMs
        if name is None:
            name = "rsmi_run"

        self.name = name
        self.dir_path = name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        # Decide on the RG procedure to implement.
        # Either RSMI, or standard majority-rule
        # (Both are examples of block-spin RG)

        if procedure == "rsmi": # TODO split into different child classes
            transform_path = os.path.join(self.dir_path, "rbms")
            self.procedure = RSMITransform(lattice_width,
                                           block_size,
                                           n_samples,
                                           name=transform_path)
            # We have to train this before we can use it
        elif procedure == "majority-rule":
            self.procedure = MajorityRuleTransform(lattice_width,
                                           block_size,
                                           n_samples)


        # Each unique value of J gets its own sequence
        def prepare_sequence(i):
            J = self.Js[i]
            sub_dir_path = os.path.join(self.dir_path, "J={}".format(J))
            return RSMISequence(J=J,
                                lattice_width=self.init_lattice_width,
                                n_samples=self.n_samples,
                                block_size=self.block_size,
                                n_steps=self.n_steps,
                                name=sub_dir_path,
                                **kwargs)

        self.rsmi_sequences = [prepare_sequence(i) for i in range(len(self.Js))]

    def train(self, n_epochs):
        step = self.get_step(0, 0)
        [v, _, ve] = step.get_load_or_create_restricted_samples()
        self.procedure.get_load_or_train_lambda_rbm(v, ve)
        self.procedure.train_lambda_rbm(v, ve, n_epochs=n_epochs)

    def gen_samples(self, overwrite=False, expectation_fns=[], mcmc="sw"):
        """ Generates samples for each sequence (i.e. runs step 0 without RG transform)
        """
        for i in range(len(self.Js)):
            @log_timer("GENERATING SAMPLES FOR SEQUENCE {} WITH J={}".format(i, self.Js[i]))
            def _run_sequence():
                self.get_sequence(i).gen_samples(overwrite=overwrite, mcmc=mcmc)

                for fn in expectation_fns:
                    self.get_sequence(i).get_expectation(fn, steps=[0])

                self.update_metadata(i)

            _run_sequence()

    def run_step(self, i, j, expectation_fns=[]):
        """
        Args:
           i (int): index of the sequence
           j (int): index of the step
        """

        # If the rbm has not yet been trained, it is
        # is trained on the first step it encounterse
        step = self.procedure.run_step(self.get_step(i, j))
        if j < self.n_steps:
            self.get_sequence(i).link_steps(j)

        return step

    def run_sequence(self, i, overwrite=False, gen_init_samples=False, expectation_fns=[]):
        """
        Args:
            i (int): index of sequence to run
            overwrite (Bool): whether to overwrite existing results
            gen_init_samples (Bool): whether to generate samples
            expectation_fns (List[Callable(samples, [mean, std])]): expectation functions to calculate

        overwrite & gen_init_samples currently have no effect --> we load all existing
        samples and trained things.
        """
        for j in range(self.n_steps):
            @log_timer("RUNNING STEP {}".format(j))
            def _run_step():
                self.run_step(i, j, expectation_fns)

            _run_step()

        self.get_expectation_of_sequence(i, expectation_fns)
        self.get_sequence(i).draw_evolution()

    def run(self, overwrite=False, gen_init_samples=True, expectation_fns=[]):
        for i in range(len(self.get_sequences())):
            @log_timer("RUNNING SEQUENCE {} WITH J={}".format(i, self.Js[i]))
            def _run_sequence():
                self.run_sequence(i, overwrite, gen_init_samples, expectation_fns)

            _run_sequence()

    def load_samples(self):
        for i, sequence in enumerate(self.get_sequences()):
            sequence.load_samples()

    def get_expectation_of_step(self, i, j, expectation_fns=[]):
        print("\n\n Expectations of seq {} step {}".format(i, j))

        seq = self.get_sequence(i)
        for fn in expectation_fns:
            seq.get_expectation(fn, steps=[j])
            self.update_metadata(i)

        return seq.metadata

    def get_expectation_of_sequence(self, i, expectation_fns=[]):
        print("\n\n Expectations of seq {}".format(i))

        seq = self.get_sequence(i)
        for fn in expectation_fns:
            seq.get_expectation(fn, steps=range(self.n_steps))
            self.update_metadata(i)

        return seq.metadata

    def get_expectations(self, expectation_fns=[], steps=None):
        if steps is None:
            steps = range(self.n_steps)

        for i, rsmi_sequence in enumerate(self.get_sequences()):
            for fn in expectation_fns:
                rsmi_sequence.get_expectation(fn, steps=steps)
                self.update_metadata(i)

    def get_step(self, i, j):
        return self.get_sequence(i).get_step(j)

    def get_sequence(self, i):
        return self.rsmi_sequences[i]

    def get_sequences(self):
        return self.rsmi_sequences

    def update_metadata(self, i):
        self.metadata[str(self.Js[i])] = { **self.metadata[str(self.Js[i])], **self.get_sequences()[i].metadata }
        print(self.metadata, "\n\n\n")
        self.write_metadata()

    def write_metadata(self, **kwargs):
        import json

        save_path = kwargs.get('save_path', 'metadata.json')
        with open("{}/{}".format(self.name, save_path), 'w+') as f:
            json.dump(self.metadata, f)
