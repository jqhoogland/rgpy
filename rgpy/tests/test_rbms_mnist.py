import os, glob
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tfrbm.bbrbm import BBRBM

n_epochs = 20
batch_size = 100
overwrite = True

name = "mnist_rbm_k=15"
filename = "{}.ckpt".format(name)
bbrbm = BBRBM(n_visible=784,
              n_hidden=128,
              k=1,
              learning_rate=0.01,
              regularization= "L1",
              lmbda=1e-3,
              use_tqdm=True)

mnist = input_data.read_data_sets('MNIST/MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

if (len(glob.glob(filename + "*")) > 0 and not overwrite):
    print("Loading weights from {}".format(filename))
    bbrbm.load_weights(filename, name)

else:
    errs = bbrbm.fit(mnist_images,
                     n_epochs=n_epochs,
                     batch_size=batch_size)
    print("saving weights to {}".format(filename))
    bbrbm.save_weights(filename, name)

    plt.plot(errs)
    plt.savefig('{}_errors.png'.format(name))


def show_digit(ax, x):
    ax.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)

def plot_reconstructions(n_images=5):
    fig, axs = plt.subplots(nrows=n_images, ncols=2, figsize=(9.3, 6))
    for i in range(n_images):
        image = mnist_images[i]
        image_rec = bbrbm.reconstruct(image.reshape(1,-1))
        ax1 = axs.flat[2 * i]
        ax2 = axs.flat[2 * i + 1]
        show_digit(ax1, image)
        show_digit(ax2,image_rec)
        ax1.set_title("{} original".format(i))
        ax2.set_title("{} reconstruction".format(i))
    plt.savefig('{}_reconstructions.png'.format(name))


def plot_gibbs_samples_from_random():
    # Gibbs samples from random starting configuration
    fig, axs = plt.subplots(nrows=10, ncols=10)
    gibbs_samples = bbrbm.gibbs_sampler()
    for gibbs_sample, ax in zip(gibbs_samples, axs.flat):
        show_digit(ax, gibbs_sample)
    plt.savefig('{}_gibbs_samples_from_random.png'.format(name))


def plot_gibbs_samples_from_example(i=0):
    # Gibbs samples from 0th training example
    fig, axs = plt.subplots(nrows=10, ncols=10)
    gibbs_samples = bbrbm.gibbs_sampler(mnist_images[i].reshape(1, -1))
    for gibbs_sample, ax in zip(gibbs_samples, axs.flat):
        show_digit(ax, gibbs_sample)
    plt.savefig('{}_gibbs_samples_from_{}.png'.format(name, i))

plot_reconstructions()
plot_gibbs_samples_from_random()
plot_gibbs_samples_from_example()
