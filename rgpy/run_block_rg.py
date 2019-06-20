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
                    n_steps=1,
                    name="rsmi_run/lw={}".format(lw),
                    procedure="majority-rule")

        rsmi.run(overwrite=True, gen_init_samples=False)
        rsmi.get_expectations(expectation_fns, steps=[0])

if __name__ == "__main__":


    # We create samples at each length scale so we can measure
    # the flow of parameters through RG
    lattice_widths = [64]# 8, ]# 32, 128]
    Js =  (CRIT_J *
           np.array([[.5, .8, .9, .97, .98, .99, 1, 1.01, 1.02, 1.1, 1.2, 1.5],
           [.5, .8, .9, .97, .98, .99, 1, 1.01, 1.02, 1.1, 1.2, 1.5],
           [.5, .8, .9, .97, .98, .99, 1, 1.01, 1.02, 1.1, 1.2, 1.5]
           ]))

    expectation_fns = ['susceptibility']

    prep_samples(lattice_widths, Js, expectation_fns)
