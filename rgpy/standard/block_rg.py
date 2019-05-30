import numpy as np

class BlockRGTransform(object):
    def __init__(self,
                 block_size):
        """ Initializes a class which performs standard RG on a given visible block.

        Args:
            n_visible: the number of visible units in a block
            binary: the kinds of units
        """
        self.block_size = block_size
        # not even used. Perhaps we could hard-code more interesting rules

    def transform(self, batch_x):
        """
        Maps a visible configuration to a hidden configuration

        Args:
           batch_x: the visible configuration to transform shape [batch_size, n_visible]
        """
        return np.sign(np.sum(2 * batch_x - 1, axis=1))

    def transform_nonbinary(self, batch_x):
        """
        Maps a visible configuration to a hidden configuration

        Args:
           batch_x: the visible configuration to transform
        """
        return np.sign(np.sum(batch_x, axis=1))
