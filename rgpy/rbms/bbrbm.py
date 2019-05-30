import tensorflow as tf
from tensorflow.train import AdamOptimizer

from rgpy.rbms import CDRBM
from rgpy.util import sample_bernoulli

class BBRBM(CDRBM):
    def get_kth_layers(self, x, y, k, is_last_sampled=tf.constant(False)):
        """
        Performs k gibbs steps (optional sampling of last visible layer)

        TODO: offer more run-time control over mean_field vs sampling
        """
        # First, we perform k-1 Gibbs steps

        cond = lambda _i, _x, _y: _i < k - 1
        def body(_i, _x, _y):
            next_x = self.get_visible(_y)
            return [_i + 1, next_x, self.get_hidden(next_x)]

        i = tf.constant(0)
        [_, x, y ] = tf.while_loop(cond,
                                   body,
                                   [i, x, y])

        # Then, we possibly bernoulli sample the last visible reconstruction
        x = self.get_visible(y, activate_x=is_last_sampled)
        y = self.get_hidden(x)

        return [x, y]

    def update_params(self, x0):
        y0 = self.get_hidden(x0)
        [xk, yk] = self.get_kth_layers(x0, y0, self.k)

        positive_grad = tf.matmul(x0, y0, transpose_a=True)
        negative_grad = tf.matmul(xk, yk, transpose_a=True)

        grad_W = (negative_grad - positive_grad) + self.grad_regularization(self.W)

        grad_vbias = (tf.reduce_sum((xk - x0), axis=0)
                       + self.grad_regularization(self.vbias))
        grad_hbias = (tf.reduce_sum((yk - y0), axis = 0)
                       + self.grad_regularization(self.hbias))

        grads = [grad_W, grad_vbias, grad_hbias]
        params = [self.W, self.vbias, self.hbias]
        grads_and_vars = (zip(grads, params))


        updates = self.optimizer.apply_gradients(grads_and_vars, self.global_step)
        # update_W = self.W.assign(self.W - self.learning_rate * delta_W)
        # update_vbias = self.vbias.assign(self.vbias - self.learning_rate *  delta_vbias)
        # update_hbias = self.hbias.assign(self.hbias- self.learning_rate *  delta_hbias)

        return updates

    def get_visible_energy(self, v):
        v_term = tf.reduce_sum(self.vbias * v)
        h_term = tf.reduce_sum(tf.log(1 + tf.exp(
            tf.matmul(tf.reshape(v, (-1, self.n_visible)), self.W) + self.hbias)))
        return -v_term - h_term

    def get_hidden_energy(self, h):
        h_term = tf.reduce_sum(self.hbias * h)
        v_term = tf.reduce_sum(tf.log(1 + tf.exp(
            tf.matmul(self.W, tf.reshape(h, (self.n_hidden, -1))) + self.vbias)))
        return -v_term - h_term

    def _initialize_vars(self):
        self.compute_hidden = self.get_hidden(self.x)
        self.compute_visible = self.get_visible(self.compute_hidden)
        self.compute_visible_from_hidden = self.get_visible(self.y)
