import _pickle as pickle
from tfrbm.samplers import *
import swendsen_wang as sw
import samplers_vanilla as vanilla
from tfrbm.visualize import draw_samples
from tfrbm.util import log_timer
J = 0.44
h = 0
lattice_width = 8
n_spins = lattice_width ** 2


def test_one_step():
    init_state = tf.constant(np.random.choice(a=[-1., 1.], size=(2, n_spins)), dtype = tf.float64)

    # Initialize the Ising Kernel (that implements state changes)
    ising = Ising1D(
        J=J,
        h=h,
        n_spins=n_spins,
    )

    next_state, _= ising.one_step(init_state, [0])
    next_next_state, _= ising.one_step(next_state, [0])

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        init_op.run()
        init_state_, next_state_, next_next_state_= sess.run([init_state, next_state, next_next_state])
        print('Init State:\n{}\n Next states:\n{}\n{}'.format(init_state_, next_state_, next_next_state_))

@log_timer("Generating Samples")
def test_ising_2d():
    # Details of configuration (1D Ising, NN)

    samples = ising_2d_generator(
        lattice_width=16,
        n_results_per_chain=1,
        n_chains=100,
        save=False)

@log_timer("Generating Samples")
def test_ising_2d_sw():
    # Details of configuration (1D Ising, NN)

    samples = sw.ising_generator(
        lattice_width=8,
        n_results_per_chain=100,
        n_chains=1,
        save=False)


@log_timer("Generating Samples")
def test_ising_2d_vanilla():
    # Details of configuration (1D Ising, NN)

    samples = vanilla.ising_generator(
        lattice_width=16,
        n_results_per_chain=5000,
        n_burnin_steps=8,
        n_steps_between_results=8,
        n_chains=1,
        save=True)


def test_expectation_fns():
    with open('crit.samples.pkl', 'rb') as f:
        samples_ = pickle.load(f)

    samples = tf.constant(samples_)
    print("Loaded {} samples of {} spins".format(*samples.shape))

    def magnetization_fn (state):
        mag = tf.reduce_mean(state)
        mag = tf.Print(mag, [tf.reshape(state, (16, 16))], "\n\nState is: ", summarize=256)
        return tf.Print(mag, [mag], "\nwith magnetization")

    def local_field(state, i):
        return tf.reduce_sum(tf.gather_nd(state,
                                          [[(i - 1) % n_spins],
                                           [(i + 1) % n_spins]]))
    def energy_fn(state):
        idxs = tf.range(0, n_spins, dtype=tf.int32)
        local_fields = tf.map_fn(lambda i: local_field(state, i), idxs, dtype=tf.float64)
        return -(h * tf.reduce_sum(state) + J * tf.reduce_sum(state * local_fields))

    def expectation(x, fn):
        return tf.reduce_mean(tf.map_fn(fn, x), axis=0)

    def expectations(x, *fns):
        return [expectation(x, fn) for fn in fns]

    mag_exp = expectation(samples, magnetization_fn)
    energy_exp = expectation(samples, energy_fn)
    combined_exp = expectations(samples, magnetization_fn, energy_fn)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        init_op.run()
        [mag_exp_, energy_exp_, combined_exp_] = sess.run([mag_exp, energy_exp, combined_exp])
        print([mag_exp_, energy_exp_, combined_exp_])


def test_generic_sampler():
    lattice_width = 10
    n_spins = lattice_width ** 2
    energy_fn = lambda config: tf.constant(1000., dtype=tf.float64) * tf.reduce_sum(config)
    sampler = GenericLatticeKernel(n_spins, energy_fn)
    init_state = tf.constant(np.random.choice(a=[-1, 1], size=(100, )), dtype=tf.float64)
    samples = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=n_spins ** 2,
        num_steps_between_results=n_spins ** 2,
        current_state=init_state,
        parallel_iterations=1,
        kernel=sampler)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        init_op.run()
        samples_= sess.run([samples])

    for i, s in enumerate(samples_[0]):
        print('\n Sample {}:\n{}'.format(i, s.reshape(lattice_width, lattice_width)))

def test_maps():
    x = tf.zeros([10, 64])
    y = tf.zeros([10, 1])
    paired_res = tf.concat([x, y], axis=1)
    paired_res = tf.Print(paired_res, [paired_res], 'paired res: ')

    def iter_fn(xy):
        prime = xy[:64] + tf.ones([64])
        energy = [xy[-1]]
        return tf.concat([prime, energy], axis=0)

    res = tf.map_fn(iter_fn, paired_res, parallel_iterations=10)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        init_op.run()
        res_, _= sess.run([res, x])

    print(res_)


def test_flatten():
    x = tf.constant([[[1, 2, 3], [2, 3, 4]], [[5, 6, 7], [8, 9, 10]]], dtype=tf.float32)

    y = tf.reshape(x, [-1, 3])
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        init_op.run()
        x_, y_= sess.run([x, y])

    print(x_, x_.shape, y_, y_.shape)

if __name__ == "__main__":
    test_ising_2d_vanilla()
