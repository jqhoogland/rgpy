from typing import List, Callable, NamedTuple, Tuple
import sys

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.train import AdamOptimizer
import numpy as np

from rgpy.util import tf_xavier_init
from rgpy.rbms import RBM
from rgpy import samplers

class RGRBM(RBM):
    def __init__(self,
                 energy_psi_of_v: Callable[[List[tf.Tensor]], tf.Tensor],
                 energy_theta_of_ve: Callable[[List[tf.Tensor], List[tf.Tensor]], tf.Tensor],
                 n_visible: int,
                 n_environment: int,
                 n_hidden: int,
                 lmbda=0,
                 regularization="L1",
                 learning_rate=0.01,
                 xavier_const=1.0,
                 use_tqdm=True,
                 # DEPRECATED:
                 tqdm=None,
                 binary=[0, 1]):
        """Initializes an RBM to implement the RSMI algorithm

        This assumes that the energy functions characterizing the (assumed)
        Boltzmann distributions P(V) and P(V,E) are provided.

        Visible bias is 0 since we care only about probability of hiddens
        conditioned on the vnisibles.

        Args:
            energy_theta_of_ve: a function that returns the energy of a state of
                visible and environment spins. Parameterized by Theta-RBM.
                Return shape [?] where ? is # of samples of v, e
            energy_psi_of_v: a function that returns the energy of a state of
                visible spins. Parametrized by Psi-RBM
                Return shape [?] where ? is # of samples of v
        """

        super(RGRBM, self).__init__(
            n_visible,
            n_hidden,
            lmbda=lmbda,
            regularization=regularization,
            learning_rate=learning_rate,
            xavier_const=xavier_const,
            use_tqdm=use_tqdm,
            # DEPRECATED:
            tqdm=tqdm,
            binary=binary
        )
        self.mutual_info = -1.
        self.n_environment = n_environment

        self.v = tf.placeholder(tf.float32, [None, self.n_visible])
        self.e = tf.placeholder(tf.float32, [None, self.n_environment])

        # 2 args: [v, e], well what's the point of the lambda function then?x
        self.energy_theta_of_ve = lambda v, e:(
            energy_theta_of_ve(v, e))

        self.energy_psi_of_v = lambda v:(
            energy_psi_of_v(v))

        # The sampler will perform the internal mc average over samples of
        # energy_theta_lambda_of_vh with clamped h
        #self.internal_mc_sampler=samplers.GenericLatticeKernel(n_spins=self.n_visible)

        # Tensorflow: Set up the computational graph
        self.optimizer = AdamOptimizer(learning_rate=self.learning_rate, epsilon=1.0)
        self.run_update_params = self.update_params(self.v, self.e)
        self._post_init()

    def _post_init(self):
        """ Initializes the tf Variables.
        """
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        # Through ? refers to the number of elements in a given batch

        # The local field (due to visible spins at each hidden site j)
        # Return shape [?, n_hidden]
        self._local_field_lambda_j=lambda v: (tf.matmul(v, self.W)) + self.hbias

        # The energy at each hidden site j = h_j* local_field at j
        # Return shape [?, n_hidden]
        self._energy_lambda_j=lambda v, h: self._local_field_lambda_j(v) * h

        # The total energy of a configuration of visible & hidden units
        # Return shape [?]
        self.energy_lambda_of_vh=lambda v, h: tf.reduce_sum(self._energy_lambda_j(v, h), axis=1)

        # The energy describing the marginal distribution of v of Boltzmann form
        # Return shape [?]
        self.energy_lambda_of_v=lambda v: (
            tf.reduce_sum(tf.log(1 + tf.exp(self._local_field_lambda_j(v))), axis=1))

        # Return shape [?]
        self.energy_psi_lambda_of_vh=lambda v, h:(
            self.energy_psi_of_v(v)
            + self.energy_lambda_of_vh(v, h)
            - self.energy_lambda_of_v(v))

        # Return shape [?]
        self.energy_theta_lambda_of_veh=lambda v, e, h: (
            self.energy_theta_of_ve(v, e)
            + self.energy_lambda_of_vh(v, h)
            - self.energy_lambda_of_v(v))

        # Return shape [?]
        self.delta_energy_psi_theta=lambda v, e: (
            self.energy_theta_of_ve(v, e) - self.energy_psi_of_v(v))

        # The gradient of energy_lambda_of_vh
        # First element: grads wrt hbias; at index j: h_j
        # Second element: grads wrt W: at index i, j: v_i * h_j
        # We neglect vbias, since this has no influence on P(h|v)
        # Return shape [[?, n_hidden],[?, n_visible, n_hidden]]
        self.delta_of_vh=lambda v, h: (
            [h, (tf.expand_dims(h, 1) * tf.expand_dims(v, 2))])

        # The gradient of energy_psi_lambda_of_vh
        # This is akin to the gradient of energy for P(h|v) of Boltzmann form
        # Return shape [[?, n_hidden],[?, n_visible, n_hidden]]
        def _grad_energy_psi_lambda_of_vh(v, h):
            h_expect = tf.nn.sigmoid(self._local_field_lambda_j(v))
            h_eff = h - h_expect
            return self.delta_of_vh(v, h_eff)

        self.grad_energy_psi_lambda_of_vh=_grad_energy_psi_lambda_of_vh

        # Grad (p_lambda(h|v)) = p_lambda(h|v) * (this term)
        # Return shape [[?, n_hidden],[?, n_visible, n_hidden]]
        def _grad_energy_lambda_of_v_not_h(v, h):
            p_lambda_h_j_given_v=tf.nn.sigmoid(self._energy_lambda_j(v, h))
            h_eff=(1 - p_lambda_h_j_given_v) * h
            return self.delta_of_vh(v, h_eff)

        self.grad_energy_lambda_of_v_not_h=_grad_energy_lambda_of_v_not_h

        self.compute_hidden = self.get_hidden(self.x)

    def gen_internal_mc_v_samples(self,
                                  E: List[tf.Tensor],
                                  h: List[tf.Tensor]):
        """
        Generates internal MC samples of V given E and H(V').

        Args:
            e: state of environmental spins
            h: state of hidden spins

        Returns:
            v_samples List[List[tf.Tensor]]: list of samples of visible spins
        """
        mc_energy_fn=lambda v: tf.reshape(self.energy_psi_lambda_of_vh(
            tf.reshape(v, [1, self.n_visible]),
            tf.reshape(h, [1, self.n_hidden])),
                                          [])

        samples = samplers.generic_generator_graph(
            mc_energy_fn,
            self.n_visible,
            n_burnin_steps=126,
            n_results_per_chain=2,
            n_chains=1,
            n_steps_between_results=126
        )

        # self.internal_mc_sampler.set_energy_fn(mc_energy_fn)

        # init_state = tf.constant(np.random.choice(a=[-1., 1.],
        #                                           size=(1, self.n_visible)),
        #                          dtype=tf.float32)
        # all_samples = tfp.mcmc.sample_chain(
        #     num_results=2,
        #     num_burnin_steps=126,
        #     num_steps_between_results=126,
        #     current_state=init_state,
        #     parallel_iterations=1,
        #     kernel=self.internal_mc_sampler)

        return samples
    def calc_internal_mc_expectation(
            self,
            e: List[tf.Tensor],
            h: List[tf.Tensor],
            expectation_fns: List[Callable[[List[tf.Tensor]], tf.Tensor]]
    ) -> List[tf.Tensor]:

        """
        Generates internal MC samples of V given E and H(V').
        Calculates expectations over these samples from functions in
        expectation_fns

        Args:
            e: state of environmental spins
            h: state of hidden spins

            expectation_fn (`V -> tf.Tensor`): an expectation function
                whose expectation we wish to calculate where `V` is a visible
                state, `E` an environmental and `H` a hidden state.
                `E` and `H` are clamped to the values e, h
        Returns:
            expectation_vals (List[tf.Tensor]): the expectation values
                corresponding to each of the functions in expectation_fns.
                shape = tf.shape(expectation_fns)
        """
        v_samples = self.gen_internal_mc_v_samples(e, h)

        return samplers.expectations(
            v_samples,
            expectation_fns)

    def expect_delta_energy(self,
                            e: List[tf.Tensor],
                            h: List[tf.Tensor]
    ):
        """Expectation value of the delta energy
        with respect to Boltzmann distribution with energy
        energy_psi_lambda_of_vh = E_{\\Psi, \\Lambda}(v, h)

        Args:
            e: state of environmental spins
            h: state of hidden spins

        Returns:
            < \\Delta E_{\\Psi, \\Theta}(v, e) >_h (h fixed)
                <=> \\Delta E_{\\Psi, \\Theta, \\Lambda}(v, e, h)
        """
        return self.calc_internal_mc_expectation(
            e,
            h,
            [lambda _v: self.delta_energy_psi_theta(_v, e)])

    def grad_mutual_info_expectations(self,
                                      e: List[tf.Tensor],
                                      h: List[tf.Tensor],
                                      debug=False):
        """Expectation value of the delta energy
        with respect to Boltzmann distribution with energy
        energy_psi_lambda_of_vh = E_{\\Psi, \\Lambda}(v, h)

        Args:
            e: state of environmental spins
            h: state of hidden spins

        Returns:
            < \\Delta E_{\\Psi, \\Theta}(v, e) >_h (h fixed)
                <=> \\Delta E_{\\Psi, \\Theta, \\Lambda}(v, e, h)
        """
        n_samples=2
        e_resized=tf.reshape(tf.tile(e[0], [n_samples]), [n_samples, self.n_environment])
        h_resized=tf.reshape(tf.tile(h[0], [n_samples]), [n_samples, self.n_hidden])

        _delta_energy_psi_theta=lambda _v: (
            self.delta_energy_psi_theta(_v, e_resized))

        _grad_energy_psi_lambda_of_vh=lambda _v: (
            self.grad_energy_psi_lambda_of_vh(_v, h_resized))

        _prod_delta_and_grad =lambda _v: (
            _grad_energy_psi_lambda_of_vh(_v))
        #            tf.reshape(_delta_energy_psi_theta(_v), [1])
        #           * _grad_energy_psi_lambda_of_vh(_v))

        (
            [delta_energy,
             [grad_h, grad_W],
             [prod_h, prod_W]]
        ) =self.calc_internal_mc_expectation(
            e, h,
            [_delta_energy_psi_theta,
             _grad_energy_psi_lambda_of_vh,
             _prod_delta_and_grad]
        )

        correct_h = lambda _h: tf.reduce_sum(_h, axis=0 )
        correct_W = lambda _W: tf.reduce_sum(_W, axis= 0)

        grad_h = correct_h(grad_h)
        grad_W = correct_W(grad_W)
        prod_h = correct_h(prod_h)
        prod_W = correct_W(prod_W)

        if debug:
            (
                [delta_energy,
                 [grad_h, grad_W],
                 [prod_h, prod_W]]
            ) = (
                [tf.Print(delta_energy, [tf.shape(delta_energy), delta_energy], "Delta E: "),
                 [tf.Print(grad_h, [tf.shape(grad_h), grad_h], "\ngrad h"),
                  tf.Print(grad_W, [tf.shape(grad_W), grad_W], "\ngrad W", summarize=10)],
                 [tf.Print(prod_h, [tf.shape(prod_h), prod_h], "\nprod h"),
                  tf.Print(prod_W, [tf.shape(prod_W), prod_W], "\n prod W", summarize=10)]]
            )
        return [delta_energy,
                [grad_h, grad_W],
                [prod_h, prod_W]]


    def grad_mutual_info_proxy(self,
                               v_prime_batch: List[List[tf.Tensor]],
                               e_batch: List[List[tf.Tensor]],
                               debug=False):
        """
        Approximates the mutual information across samples of X (the entire
        configuration).

        Indices should match, so e_samples[i] is the environmental block
        corresponding to v_samples[i]

        Args:
            v_prime_batch: batch of visible samples from external mc
                Shape: [batch_size, n_visible]
            e_batch: batch of environemtnal samples from external mc
                Shape: [batch_size, n_environment]
            batch_size:

        Returns:
            The gradient of a mutualinformation proxy to be used for
            differentiation in SGD.
            < \\Delta E_{\\Psi, \\Theta}(v, e) >_h (h fixed)
                where h = h(v) from v_samples, e = e from e_samples
                and internal v MC sampled
        """
        batch_size = tf.shape(v_prime_batch)[0]

        # Shape: [batch_size, n_hidden]
        h_batch = self.get_hidden(v_prime_batch)

        # Shape: [[batch_size, n_hidden], [batch_size, n_visible, n_hidden]]
        delta_v_prime_not_h=(
            lambda v, h: self.grad_energy_lambda_of_v_not_h(v, h))

        grad_mutual_info_per_term = (
            lambda delta_cond, delta_energy, prod, delta_not: (
                -delta_energy * (delta_not + delta_cond) + prod))

        def compute_term(a: tf.Tensor):
            """
            Computes the ath term of the external sum in grad_mutual_info_proxy
            TODO: Parallelize this; i.e. allow joint calculation for entire batches
            """
            correct_shape = lambda x, x_len: tf.reshape(x, [-1, x_len])
            v_prime= correct_shape(v_prime_batch[a],self.n_visible)
            e= correct_shape(e_batch[a],self.n_environment)
            h= correct_shape(h_batch[a],self.n_hidden)

            (
                [delta_energy,
                 [h_grad, W_grad],
                 [h_prod, W_prod]]
            ) =self.grad_mutual_info_expectations(e, h)

            [h_grad_not, W_grad_not] = delta_v_prime_not_h(v_prime, h)

            h_term=tf.reshape(
                grad_mutual_info_per_term(h_grad, delta_energy, h_grad, h_grad_not),
                [1])


            W_term=tf.reshape(
                grad_mutual_info_per_term(W_grad, delta_energy, W_prod, W_grad_not),
                [4, 1])
            mi_term = tf.reshape(-delta_energy, [])

            if debug:
                [h_term, W_term, mi_term] = (
                    [tf.Print(h_term, [tf.shape(h_term), h_term], "\nh_term"),
                     tf.Print(W_term, [tf.shape(W_term), W_term], "\nW term", summarize=100),
                     tf.Print(mi_term, [tf.shape(mi_term), mi_term], "\n Mutual info proxy")]
                )

            return [h_term, W_term, mi_term]

        i = tf.constant(0)
        h_sum_init = tf.zeros([self.n_hidden], dtype=tf.float32)
        W_sum_init = tf.zeros([self.n_visible, self.n_hidden], dtype=tf.float32)
        mi_sum_init = tf.constant(0., dtype=tf.float32)

        def body(_i, _h_sum, _W_sum, _mi_sum):
            [_h_update, _W_update, _mi_update] = compute_term(_i)
            return [_i + 1,
                    _h_sum + _h_update,
                    _W_sum + _W_update,
                    _mi_sum + _mi_update]

        cond = lambda _i, _h_sum, _W_sum, _mi_sum: _i < batch_size

        [_, h_sum, W_sum, mi_sum] = tf.while_loop(cond,
                                                  body,
                                                  [i, h_sum_init, W_sum_init, mi_sum_init])

        return [h_sum, W_sum, mi_sum]

    def update_params(self, v, e):
        [*grads, mutual_info] = self.grad_mutual_info_proxy(v, e)
        params = [self.hbias, self.W]
        update_op = self.optimizer.apply_gradients(zip(grads, params), self.global_step)
        return update_op, mutual_info

    def partial_fit(self, batch_ve):
        batch_v, batch_e = batch_ve
        update_op, mutual_info= self.run_update_params
        _, mutual_info_= self.sess.run([update_op, mutual_info],
                                       feed_dict={self.v: batch_v, self.e: batch_e})
        return mutual_info_

    def mutual_info_proxy(self):
        """
        Approximates a mutual information proxy across samples of X (the entire
        configuration).

        Indices should match, so e_samples[i] is the environmental block
        corresponding to v_samples[i]

        Returns:
            An approximation of the mutual information proxy
                < \\Delta E_{\\Psi, \\Theta}(v, e) >_h (h fixed)
                where h = h(v) from v_samples, e = e from e_samples
                and internal v MC sampled
                Shape= []
        """
        h_samples = self.get_hidden(self.v_samples)
        tf_zip = lambda a, b: tf.stack([a, b], 1)
        double_map_fn = lambda map_fn: (lambda ab: map_fn(a[0], a[1]))

        expect_delta_energy = tf.map_fn(
            double_map_fn(self.expect_delta_energy),
            tf_zip(self.e_samples, h_samples))
        return T.mean(expect_delta_energies)

    def fit(self,
            data_ve,
            n_epochs=10,
            batch_size=10,
            print_filters_dir="./",
            shuffle=True,
            verbose=True):
        """
        Trains the Lambda-RBM. Performs SGD where the derivative is given by
        self.grad_mutual_info_proxy.

        Args:
            `data_ve`: training data [ve_samples]
            `n_epochs`: number of epochs to train over
            `batch_size`: number of training samples per batch (external mc-avg)
            `shuffle`: whether to shuffle training data before fitting
            `verbose`: whether to display information about training
        """
        assert n_epochs > 0

        data_ve = self.convert_to_binary(data_ve)

        n_data = min(data_ve.shape[0], 20000)

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1


        batch_spacing = n_data / n_batches

        errs = []

        for e in range(n_epochs):
            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                inds = np.arange(n_data)
                np.random.shuffle(inds)
                data_ve = data_ve[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_ve = data_ve[b * batch_size:(b + 1) * batch_size]

                # TODO: avoid splitting this up here
                batch_v = batch_ve[:, :self.n_visible]
                batch_e = batch_ve[:, self.n_visible:]

                self.mutual_info = self.partial_fit([batch_v, batch_e])
                epoch_errs[epoch_errs_ptr] = -self.mutual_info # <=> batch_error
                epoch_errs_ptr += 1

            if verbose:
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
