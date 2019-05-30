import os, math, timeit, logging
try:
    import CPickle as pickle
except ImportError:
    import pickle

import matplotlib.pyplot as plt

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_probability as tfp

import tfrbm.samplers as tf_sample
from tfrbm.bbrbm import BBRBM
from tfrbm import visualize

path_list = ['cold' ,'hot', 'belowcrit']
learning_rate = 5 * 1e-3
N = 40
full_n_hidden = 128
 n_epochs = 4000
lmbda = 1e-4
batch_size = 100
k = 10
overwrite = False
n_hidden = 5

def get_ve_samples(path):
    save_path = "{}.restricted.pkl".format(path)
    # Load the file if restrictions have already been processed
    if os.path.isfile(save_path):
        print("{}\t Loading restricted samples from {}".format(path, save_path))
        with open(save_path, 'rb') as f:
            res = pickle.load(f)
    else:
        raise OSError('Could not find {}'.format(save_path))
    return res

@pytest.fixture
def psi_rbm_prepare(path, draw_chains=True):
    """ Prepares psi samples and trains psi rbm for E(v) energy function
    Returns:
        v_train (): The training set
        psi_rbm
    """
    save_path = "psi_{}.ckpt".format(path)
    v_samples, _, _= get_ve_samples(path)
    print("{}\t Dividing into training/testing set".format(path))
    v_train, v_test, _, _= train_test_split(v_samples,
                                            np.zeros(v_samples.shape[0]),
                                            random_state=123,
                                            shuffle=True)

    psi_rbm = BBRBM(
        learning_rate=learning_rate,
        n_visible=4,
        n_hidden=n_hidden,
        lmbda=lmbda,
        k=k,
        use_tqdm=True
    )
    if (os.path.isfile(save_path + ".meta") and not overwrite):
        print("{}\t Loading RBM from {}".format(path, save_path))

        psi_rbm.load_weights(save_path, path)
    else:


        print("{}\t Training Psi RBM".format(path))
        start_time = timeit.default_timer()

        psi_rbm.fit(v_train,
                    n_epochs=n_epochs,
                    batch_size=batch_size)


        end_time = timeit.default_timer()
        time = (end_time - start_time) / 60
        print("{}\t Finished Training Psi RBM. Took {} minutes".format(path, time))
        print("{}\t Saving training/testing sets and RBM to {}".format(path, save_path))
        psi_rbm.save_weights(save_path, path)
    return v_train, psi_rbm

def show_im(ax, x, N):
    ax.imshow(x.reshape((N, N)), cmap=plt.cm.gray)

def plot_reconstructions(bbrbm, samples, name, n_images=10, N=2):
    fig, axs = plt.subplots(nrows=n_images, ncols=2, figsize=(9.3, 6))
    for i in range(n_images):
        image = samples[i]
        image_rec = bbrbm.reconstruct(image.reshape(1,-1))
        ax1 = axs.flat[2 * i]
        ax2 = axs.flat[2 * i + 1]
        show_im(ax1, image, N)
        show_im(ax2,image_rec, N)
        ax1.set_title("{} original".format(i))
        ax2.set_title("{} reconstruction".format(i))
        plt.savefig('{}_reconstructions.png'.format(name))


def plot_gibbs_samples_from_random(bbrbm, name, N=2):
    # Gibbs samples from random starting configuration
    fig, axs = plt.subplots(nrows=10, ncols=1)
    gibbs_samples = bbrbm.gibbs_sampler(n_samples=10)
    for gibbs_sample, ax in zip(gibbs_samples, axs.flat):
        show_im(ax, gibbs_sample, N)
        plt.savefig('{}_gibbs_samples_from_random.png'.format(name))


def plot_gibbs_samples_from_example(bbrbm,samples, name, i=0, N=2):
    # Gibbs samples from 0th training example
    fig, axs = plt.subplots(nrows=10, ncols=1)
    gibbs_samples = bbrbm.gibbs_sampler(samples[i].reshape(1, -1), n_samples=10)
    for gibbs_sample, ax in zip(gibbs_samples, axs.flat):
        show_im(ax, gibbs_sample, N)
        plt.savefig('{}_gibbs_samples_from_{}.png'.format(name, i))

@pytest.fixture
def full_config_prepare():
    # with open('crit.samples.pkl','rb') as f:
    #    crit_samples = pickle.load(f)

    with open("Ising2DFM_reSample_L40_T=2.25.pkl", 'rb') as f:
        crit_samples = pickle.load(f)
        crit_samples = np.unpackbits(crit_samples).reshape(-1, 1600).astype('int')

    full_rbm = BBRBM(
        n_visible=N ** 2,
        n_hidden=full_n_hidden,
        regularization="L1",
        lmbda=lmbda,
        k=k,
        use_tqdm=True
    )
    name = 'crit.full'
    save_path = '{}.ckpt'.format(name)

    if os.path.isfile(save_path) and not overwrite:
        print("Loading full RBM")
        full_rbm.load_weights(save_path)
    else:
        print("Training full RBM")
        full_rbm.fit(crit_samples,
                     n_epochs=n_epochs,
                     batch_size=batch_size)

        full_rbm.save_weights(save_path, name)

    return crit_samples, full_rbm, name

def test_rbm_full_config(full_config_prepare):
    crit_samples, full_rbm, name = full_config_prepare

    # print("plotting reconstructions, gibbs evolution (from random and from givenstate)")
    # plot_reconstructions(psi_rbm, crit_samples, path)
    # plot_gibbs_samples_from_random(full_rbm, path)
    # plot_gibbs_samples_from_example(full_rbm, crit_samples, path)

    path = "{}.png".format(name)
    plot_reconstructions(full_rbm, crit_samples, path, N=N)
    plot_gibbs_samples_from_random(full_rbm, path, N=N)
    plot_gibbs_samples_from_example(full_rbm, crit_samples, path, N=N)

    [W, vbias, hbias] = full_rbm.get_weights()

    #vbias = tf.abs(vbias) * tf.constant(1000.)
    #hbias = tf.abs(hbias) * tf.constant(1000.)

    full_rbm_kernel = tf_sample.RBMKernel(W, vbias, hbias)

    print("{}\t Sampling Full RBM".format(name))

    init_state = np.random.choice(a=[-1., 1.], size=(int(N ** 2), ))

    samples = tfp.mcmc.sample_chain(
        num_results=100,
        num_burnin_steps=N ** 2,
        num_steps_between_results=N ** 2,
        current_state=init_state,
        parallel_iterations=1,
        kernel=full_rbm_kernel)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        init_op.run()
        samples_= sess.run([samples])

    print("{}\t Drawing Full RBM Samples".format(name))

    visualize.draw_samples(np.array(samples_[0][0]), (N, N), (10, 10),
                           path='./full_{}_mc_samples.png'.format(name))


@pytest.mark.parametrize('path', path_list)
def test_psi_rbm(path, psi_rbm_prepare):
    v_train, psi_rbm = psi_rbm_prepare

    n_visible = 4
    lattice_width = 2

    print("plotting reconstructions, gibbs evolution (from random and from givenstate)")
    plot_reconstructions(psi_rbm, v_train, path)
    plot_gibbs_samples_from_random(psi_rbm, path)
    plot_gibbs_samples_from_example(psi_rbm, v_train, path)

    [W, vbias, hbias] = psi_rbm.get_weights()

    psi_rbm_kernel = tf_sample.RBMKernel(W, vbias, hbias)

    print("{}\t Sampling Psi RBM".format(path))

    init_state = np.random.choice(a=[-1., 1.], size=(n_visible, ))

    samples = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=126,
        num_steps_between_results=100,
        current_state=init_state,
        parallel_iterations=1,
        kernel=psi_rbm_kernel)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        init_op.run()
        samples_= sess.run([samples])

    print("{}\t Drawing Psi RBM Samples".format(path))

    visualize.draw_samples(np.array(samples_[0][0]), (2, 2), (20, 50),
                           path='./psi_{}_mc_samples.png'.format(path))

if __name__ == "__main__":
    test_rbm_full_config(full_config_prepare())
