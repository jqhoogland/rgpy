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

        #rsmi.train(n_epochs=10)
        rsmi.run(overwrite=False)
        rsmi.get_expectations(expectation_fns)

if __name__ == "__main__":


    # We create samples at each length scale so we can measure
    # the flow of parameters through RG
    lattice_widths = [64]
    Js =  (CRIT_J *
           np.array([[1, .97, .98, .99, 1.01, 1.02, 1.03]]))

    expectation_fns = ['magnetization', 'nn_correlation', 'nnn_correlation', 'susceptibility']

    prep_samples(lattice_widths, Js, expectation_fns)
