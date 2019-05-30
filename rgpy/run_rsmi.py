import math, os

from rsmi_renormalization.rsmi import *

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
                    n_steps=max(i, 1),
                    name="rsmi_run/lw={}".format(lw))

        #rsmi.gen_samples(overwrite=True, mcmc="sw")
        rsmi.get_expectations(expectation_fns)

if __name__ == "__main__":


    # We create samples at each length scale so we can measure
    # the flow of parameters through RG
    lattice_widths = [8, 16, 32, 64, 128]
    Js =  (CRIT_J *
           np.array([[.0625, .125, .25, .5, .8, .9, 1, 1.1, 1.2, 1.5, 2, 4, 8],
                     [.0625, .125, .25, .5, .8, .9, 1, 1.1, 1.2, 1.5, 2, 4, 8],
                     [.0625, .125, .25, .5, .8, .9, 1, 1.1, 1.2, 1.5, 2, 4, 8],
                     [.0625, .125, .25, .5, .8, .9, 1, 1.1, 1.2, 1.5, 2, 4, 8],
                     [.0625, .125, .25, .5, .8, .9, 1, 1.1, 1.2, 1.5, 2, 4, 8]]))

    expectation_fns = ['magnetization', 'nn_correlation', 'nnn_correlation']

    prep_samples(lattice_widths[4:], Js[4:], expectation_fns)
