import os
from tfrbm.bbrbm import BBRBM
from tfrbm import visualize
try:
    import _pickle as pickle
except:
    import pickle

sample_descriptors = ['cold', ]

def get_samples(sample_descriptor):
    filepath = './crit.samples.pkl'
    # Load the file if restrictions have already been processed
    with open(filepath, 'rb') as f:
        samples = pickle.load(f)
    return samples

lattice_width = 16
n_visible = 256

samples = get_samples(sample_descriptors[0])
visualize.draw_samples(samples, (lattice_width, lattice_width), (25, 40),
                       'crit.samples.png')
bbrbm = BBRBM(n_visible=n_visible, n_hidden=128, learning_rate=0.01, use_tqdm=True)
errs = bbrbm.fit(samples, n_epoches=100, batch_size=10)

new_samples = bbrbm.gibbs_sampler()
visualize.draw_samples(new_samples, (lattice_width, lattice_width), (25, 40),
                       'crit.gibbs_chains.png')
