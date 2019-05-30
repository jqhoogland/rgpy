import math, os
import tensorflow as tf
from tensorflow.train import AdamOptimizer

import numpy as np
import sys
from rsmi_renormalization.util import tf_xavier_init, sample_bernoulli

from rsmi_renormalization import visualize

class RBM (object):
    def __init__(self,
                 n_visible,
                 n_hidden,
                 lmbda=0,
                 regularization="L1",
                 learning_rate=0.01,
                 xavier_const=1.0,
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None,
                 binary=[0, 1]):

        # Logging progress during training (in real time)
        self._use_tqdm = use_tqdm
        self._tqdm = None
        self.binary = binary

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_visible_rt = int((math.sqrt(n_visible)))
        self.n_hidden = n_hidden
        self.n_hidden_rt = int((math.sqrt(n_hidden)))
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0)

        # Regularization: Can be one of L1 (not yet implemented), L2
        self.lmbda = lmbda

        grad_regularization = lambda x: 0.
        if regularization == "L1":
            grad_regularization = lambda x: self.lmbda * tf.sign(x)

        elif regularization == "L2":
            grad_regularization = lambda x: self.lmbda * tf.abs(x)
        else:
            raise ValueError('{} regularization not recognized'
                             .format(regularization))

        self.grad_regularization = grad_regularization
        # Layers: Visible and Hidden
        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        # Weights: Couplings and Biases
        self.W = tf.Variable(
            tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const),
            dtype=tf.float32)
        self.vbias = tf.Variable(tf.zeros([self.n_visible]),
                                 dtype=tf.float32)
        self.hbias = tf.Variable(tf.zeros([self.n_hidden]),
                                 dtype=tf.float32)

        # Implementations depend on type of RBM (BBRBM vs others)
        self._initialize_vars()

    def _post_init(self):
        """ Initializes the tf Variables.
        """
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        """
        Initializes:
            self.update_weights
            self.update_deltas
            self.compute_hidden
            self.compute_visible
            self.compute_visible_from_hidden
        """
        raise NotImplementedError

    def get_visible(self, y,
                    activate_y=tf.constant(False),
                    activate_x=tf.constant(False)):

        y = tf.cond(activate_y,
                    lambda: sample_bernoulli(y),
                    lambda: y)
        x = tf.nn.sigmoid(
            tf.matmul(y, tf.transpose(self.W))
            + self.vbias)
        x = tf.cond(activate_x,
                    lambda: sample_bernoulli(x),
                    lambda: x)
        return x


    def get_hidden(self,
                   x,
                   activate_x=tf.constant(False),
                   activate_y=tf.constant(False)):

        x = tf.cond(activate_x,
                    lambda: sample_bernoulli(x),
                    lambda: x)

        y = tf.nn.sigmoid(tf.matmul(x, self.W) + self.hbias)

        y = tf.cond(activate_y,
                    lambda: sample_bernoulli(y),
                    lambda: y)
        return y


    def get_err(self, batch_x: np.array):
        """
        Computes the error (mean-squared or cosine) between an inputted visible
        configuration and its reconstruction
        """
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        """
        Computes the free energy. Must be implemented in child classes.
        """
        raise NotImplementedError

    def transform(self, batch_x: np.array, activate=False):
        """
        Maps a visible configuration to a hidden configuration

        Args:
           batch_x: the visible configuration to transform
        """
        hidden = self.get_hidden(self.x, activate_y=tf.constant(activate))
        return self.sess.run(hidden,
                             feed_dict={self.x: batch_x})

    def transform_nonbinary(self, batch_x: np.array, activate=True):
        """
        Maps a visible configuration to a hidden configuration

        Args:
           batch_x: the visible configuration to transform
        """
        batch_x = self.convert_to_binary(batch_x)
        hidden = self.get_hidden(self.x, activate_y=tf.constant(activate))
        hidden_ = self.sess.run(hidden,
                             feed_dict={self.x: batch_x})
        return self.convert_from_binary(hidden_)


    def partial_fit(self, batch_x):
        self.sess.run(self.run_update_params, feed_dict={self.x: batch_x})

    def draw_filters(self, save_path):
        [W, _, _] = self.get_weights()
        print("Drawing filters to: {}".format(save_path))
        visualize.draw_samples(W.transpose(),
                               (self.n_hidden_rt, self.n_hidden_rt),
                               save_path)


    def convert_to_binary(self, data):
        print("conveting to binary", self.binary)
        return np.where(data == self.binary[0], 0, 1)

    def convert_from_binary(self, data):
        return np.where(data == 0, *self.binary)

    def fit(self,
            data_x:np.array,
            n_epochs=10,
            batch_size=10,
            shuffle=True,
            verbose=True,
            print_filters_dir="./",
            print_filters_every=5):
        """
        Fits the RBM on training data data_x

        Args:
            data_x: training data
            n_epochs: number of epochs to train over
            batch_size: number of training samples per batch
            shuffle: whether to shuffle training data before fitting
            verbose: whether to display information about training
        """
        assert n_epochs > 0

        # If inputted data is not in [0,1], map it to [0,1]

        print("before converting to binary", data_x[0:2])
        data_x = self.convert_to_binary(data_x)
        print('after', data_x[0:2])

        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        errs = []

        if verbose:
            self.draw_filters(os.path.join(print_filters_dir,
                                           "filters_at_epoch_0.png"))

        for e in range(n_epochs):

            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                inds = np.arange(n_data)
                np.random.shuffle(inds)
                data_x = data_x[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:

                if  ((e + 1) % print_filters_every) == 0:
                    self.draw_filters(os.path.join(print_filters_dir,
                                                   "filters_at_epoch_{}.png".format(e + 1)))

                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                    sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def get_weights(self):
        """
        Returns:
            self.W (np.array): the couplings between visible and hidden layers
            self.vbias (np.array): the visible layer biases
            self.hbias (np.array): the hidden layer biases
        """
        return (self.sess.run(self.W),
                self.sess.run(self.vbias),
                self.sess.run(self.hbias))

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_W': self.W,
                                name + '_vbias': self.vbias,
                                name + '_hbias': self.hbias})
        return saver.save(self.sess, filename)

    def set_weights(self, W, vbias, hbias):
        self.sess.run(self.W.assign(W))
        self.sess.run(self.vbias.assign(vbias))
        self.sess.run(self.hbias.assign(hbias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_W': self.W,
                                name + '_vbias': self.vbias,
                                name + '_hbias': self.hbias})
        saver.restore(self.sess, filename)

    def gibbs_sampler(self, visibles=None, n_samples=100, n_steps_between_samples=1):
        """
        Runs a gibbs chain from initial state visibles.

        Returns:
             visibles: the state after `n_steps` gibbs steps
                 (default: None (random state))
        """
        if visibles is None:
            visibles = np.random.choice(a=[-1., 1], size=(1, self.n_visible))

        v_samples = np.zeros((n_samples, self.n_visible))
        v_samples[0] = visibles
        n_steps = n_samples * n_steps_between_samples

        for i, step in enumerate(range(1, n_steps)):
            visibles = self.reconstruct(visibles)
            j = i % n_steps_between_samples
            if j == 0:
                v_samples[i // n_steps_between_samples + 1] = visibles

        return v_samples

class CDRBM(RBM):
    def __init__(self,
                 n_visible,
                 n_hidden,
                 lmbda=0,
                 regularization="L2",
                 learning_rate=0.01,
                 xavier_const=1.0,
                 err_function='mse',
                 k=1,
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None,
                 binary=[0, 1]):
        self.k = k
        RBM.__init__(self,
                     n_visible,
                     n_hidden,
                     lmbda,
                     regularization,
                     learning_rate,
                     xavier_const,
                     use_tqdm,
                     tqdm,
                     binary=binary)
        # Error Functions
        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(
                tf.square(self.x - self.compute_visible))

        # Tensorflow: Set up the computational graph
        self.optimizer = AdamOptimizer(learning_rate=self.learning_rate, epsilon=1.0)
        self.run_update_params = self.update_params(self.x)
        self._post_init()

    def transform_inv(self, batch_y: np.array):
        """
        Maps a hidden configuration to a visible configuration

        Args:
            batch_y: the hidden configuration to transform
        """
        return self.sess.run(self.compute_visible_from_hidden,
                             feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x: np.array):
        """
        Maps a visible configuration to a hidden configuration and back.

        Args:
            batch_x: the visible configuration to reconstruct
        """
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})
