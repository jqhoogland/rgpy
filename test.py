import math, os

from rgpy.rsmi import *

CRIT_J = math.log(1 + math.sqrt(2)) / 2

def prep_samples(lattice_widths, Js, expectation_fns):
    """
    Prepares samples and measures expectations (to store in metadata)
    """
    if not os.path.isdir("rsmi_run"):
        os.mkdir("./rsmi_run")

    stop = 1
    for i in range(len(lattice_widths)):
        lw = lattice_widths[i]
        rsmi = RSMI(Js=Js[i],
                    lattice_width=lw,
                    n_samples=1000,
                    n_steps=3,
                    name="rsmi_run/lw={}".format(lw))

        rsmi.procedure.load_lambda_rbm()
        # _W=np.ones([4, 1]) * 100 + np.random.uniform([4, 1]) * 10

        # W = tf.constant(_W, dtype=tf.float32)
        # hbias = tf.constant([2 * np.mean(_W)], dtype=tf.float32)

        _W=np.random.normal(104, 5, size=[4, 1])
        #W = tf.constant([[100.], [100.], [100.], [100.]], dtype=tf.float32)
        hbias = tf.constant([-202.42462334], dtype=tf.float32)

        W = tf.constant(_W, dtype=tf.float32)
        # hbias = tf.constant([2 * np.mean(_W)], dtype=tf.float32)


        rsmi.procedure.lambda_rbm.W=W
        rsmi.procedure.lambda_rbm.hbias=hbias

        rsmi.run(overwrite=False)
        rsmi.get_expectations(expectation_fns)

if __name__ == "__main__":


    # We create samples at each length scale so we can measure
    # the flow of parameters through RG
    lattice_widths = [64]
    Js =  (CRIT_J *
           np.array([[.96]]))

    expectation_fns = ['magnetization', 'nn_correlation', 'nnn_correlation', 'susceptibility']

    prep_samples(lattice_widths, Js, expectation_fns)
