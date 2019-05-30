import math
import _pickle as pickle
from rsmi_renormalization.rsmi import *
from rsmi_renormalization.samplers import *

crit_J = math.log(1 + math.sqrt(2)) / 2
Js_unitless = np.arange(2, 5, 0.5)
Js = Js_unitless * crit_J

rsmi = RSMI(Js=Js,
             lattice_width=32,
             n_samples=1000,
             n_steps=1,
             name="rsmi_run_test_5"
)

expectation_fns = ['magnetization', 'nn_correlation', 'nnn_correlation']

#crit1.gen_samples(overwrite=True, expectation_fns=expectation_fns)

# step0 = crit1.rsmi_sequences[0].rsmi_steps[0]
# step0.gen_restricted_samples()
# step0.save_restricted_samples()

# crit1.load_samples(expectation_fns=expectation_fns)
# print("before ", step0.get_restricted_samples())
# step0.load_restricted_samples()
# print("after ", step0.get_restricted_samples())

rsmi.run(overwrite=True, gen_init_samples=False, expectation_fns=expectation_fns)

#rsmi.get_expectations(expectation_fns)

#seq1 = RSMISequence(J= 0.44, lattice_width=8, n_samples=10, n_steps=2)
#seq1.run(overwrite=True)

# def load_data_pkl(J):
#     with open("./rsmi_run/J={}/0/samples/samples.x.pkl".format(J), 'rb') as f:
#         return pickle.load(f)
