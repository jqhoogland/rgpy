import timeit
import numpy as np
import tensorflow as tf
import scipy

def tf_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)


def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)

def save_images(images, size, path):
    """
    Save the samples images
    The best size number is
            int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
    example:
        The batch_size is 64, then the size is recommended [8, 8]
        The batch_size is 32, then the size is recommended [6, 6]
    """
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    merge_img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w] = image

    return scipy.misc.imsave(path, merge_img)

def log_timer(str):
    def log_timer_generator(fn):
        def new_fn(*args, **kwargs):
            print("\n--------------------------------------------------------------------------------\n\nStarted: {}\n".format(str))
            start_time = timeit.default_timer()
            res = fn(*args, **kwargs)
            end_time = timeit.default_timer()
            run_time = (end_time - start_time) / 60.
            print ("\nCompleted: {}\nTook {} minutes\n\n--------------------------------------------------------------------------------\n\n".format(str, run_time))
            return res
        return new_fn
    return log_timer_generator
