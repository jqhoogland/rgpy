import os

import tensorflow as tf

from tfrbm.rsmi_step import *
from tfrbm.samplers import *

dir_path = "rsmi_run"

get_samples = lambda J, i, kind: os.path.join(
    dir_path, "J={}/{}/samples/samples.{}.pkl".format(J, i, kind))

calculator = Expectation(
    lattice_width=16,
)

n_steps = 3

def get_expectations(samples):
    return {'Magnetization': calculator.get_magnetization(samples),
            "Nearest Neighbor Correlations": calculator.get_nn_correlation(samples),
            "Next Nearest Neighbor Correlations": calculator.get_nnn_correlation(samples)}

def load_x_samples(load_path):
    with open(load_path, 'rb') as f:
        samples = pickle.load(f)
    return samples

for J in [0.43, 0.44, 0.45, 0.46]:
    print("\n\nJ= ", J)
    calculator.J = J
    calculator.set_lattice_width(16)
    samples = load_x_samples(get_samples(J, 0, 'x'))
    print(get_expectations(samples))

    for i in range(n_steps):
        calculator.set_lattice_width(8 // (2 ** i))
        samples = load_x_samples(get_samples(J, 0, 'x'))
        print(get_expectations(samples))
