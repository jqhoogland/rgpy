import numpy as np
import tensorflow as tf
from tfrbm.util import sample_bernoulli

n_visible = 5
n_hidden = 2

X = tf.constant([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], dtype=tf.float32)
k = 2

# Weights: Couplings and Biases
W = tf.Variable(
    tf.random.uniform([n_visible, n_hidden],
                      dtype=tf.float32))
vbias = tf.Variable(tf.zeros([n_visible]),
                    dtype=tf.float32)
hbias = tf.Variable(tf.zeros([n_hidden]),
                    dtype=tf.float32)

grad_regularization = lambda x: 0.001 * tf.sign(x)
learning_rate = 0.001

def get_hidden(x, is_sampled=tf.constant(False)):
    # Shape: [?, n_visible]
    x = tf.cond(is_sampled,
                lambda: sample_bernoulli(x),
                lambda: x)
    return tf.nn.sigmoid(tf.matmul(x, W) + hbias)

def get_visible(y, is_sampled=tf.constant(False)):
    y = tf.cond(is_sampled,
                lambda: sample_bernoulli(y),
                lambda: y)
    return tf.nn.sigmoid(
        tf.matmul(y, tf.transpose(W))
        + vbias)

def get_kth_layers(x, y, k):
    i = tf.constant(0)
    cond = lambda i, x, y: i < k
    def body(i, x, y):
        next_x = get_visible(y)
        return [i + 1, next_x, get_hidden(next_x)]

    [_, x, y ] = tf.while_loop(cond,
                               body,
                               [i, x, y])

    return [x, y]


hidden_p = get_hidden(X)
visible_recon_p, hidden_recon_p = get_kth_layers(X, hidden_p, k)

positive_grad = tf.matmul(X, hidden_p, transpose_a=True)
negative_grad = tf.matmul(visible_recon_p, hidden_recon_p, transpose_a=True)

delta_W = (negative_grad - positive_grad) + grad_regularization(W)

delta_vbias = tf.reduce_sum((visible_recon_p - X), axis=0) + grad_regularization(vbias)
delta_hbias = tf.reduce_sum(hidden_recon_p - hidden_p, axis = 0)+ grad_regularization(hbias)

update_W = (W - learning_rate * delta_W)
update_vbias = (vbias - learning_rate *  delta_vbias)
update_hbias = (hbias- learning_rate *  delta_hbias)


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    (
        [X_, hidden_p_, visible_recon_p_, hidden_recon_p_]
    )= sess.run([X,hidden_p, visible_recon_p, hidden_recon_p])

    print('visible 0\n{}\n hidden 0\n{}\nvisible k\n{}\n hidden k\n{}'
          .format(X_, hidden_p_, visible_recon_p_, hidden_recon_p_))

    (
        [W_, vbias_, hbias_]
    ) = sess.run([W, vbias, hbias])

    print("\n\nORIGINALS:\nW:\n{}\nv:\n{}\nh:\n{}".format(W_, vbias_, hbias_))


    (
        [delta_W_, delta_vbias_, delta_hbias_]
    ) = sess.run([delta_W, delta_vbias, delta_hbias])

    print("\n\nDELTAS:\nW:\n{}\nv:\n{}\nh:\n{}".format(delta_W_, delta_vbias_, delta_hbias_))

    (
        [update_W_, update_vbias_, update_hbias_]
    ) = sess.run([update_W, update_vbias, update_hbias])

    print("\n\nUPDATES:\nW:\n{}\nv:\n{}\nh:\n{}".format(update_W_, update_vbias_, update_hbias_))
