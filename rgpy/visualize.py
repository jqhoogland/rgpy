import math
try:
    import PIL.Image as Image
except:
    import Image

import numpy as np

from DeepLearningTutorials.code.utils import tile_raster_images

def extend_to_perfect_square(vector):
    # Takes a vector and appends empty entries so as to
    # achieve a square [?, vector length]
    vector = np.asarray(vector)
    #print(vector.shape)

    n_vectors, length = vector.shape
    rounded_sqrt = int(math.sqrt(length))
    perf_sq = rounded_sqrt ** 2

    if perf_sq > length:
        diff = perf_sq - length
        vector =np.concatenate([vector,np.zeros((n_vectors, diff))],axis=1)

    if perf_sq < length:
        vector = vector[:, :perf_sq]

    img_shape = (rounded_sqrt, rounded_sqrt)
    return vector, img_shape


def draw_samples(samples, tile_shape, path='samples.png'):
    """Draws all samples on grid according to provided shape

        Args:
            img_shape (tuple of ints): the shape of the image to be drawn
                Assumed to match size of samples
            **kwargs: containing possible path to the target directory or file
        """
    # the path_to_filename_path decorator handles defaults
    X, img_shape = extend_to_perfect_square(samples)
    image = Image.fromarray(
        tile_raster_images(
            X=X,
            img_shape=img_shape,
            tile_shape=tile_shape,
            tile_spacing=(1, 1),
            scale_rows_to_unit_interval=False,

            )
    )
    image.save(path)
