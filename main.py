import numpy as np
import skimage
import warnings
import random
def generate_multilabel_toy_dataset(sample_number=1000, x_res=256, y_res=256, channels=3, v_min=0, v_max=1,
                                      label_count=4, label_frequency=0.5, path=None, save_to_folder = False):
    """
    Creates a dataset with basic shapes; 1, 2, 3, 4... = circle, line, triangle, square etc...

    sample_number   - number of samples to generate
    x_res, y_res    - dimensions of generated image
    channels        - number of image channels to use
    v_min, v_max    - minimum and maximum values used in channels (typically 0-1 for samples).

    label_count     - number of classes to generate dataset with
    label_frequency - how frequently to have a label occur in an image
    path            - where to save the dataset to
    save_to_folder  - if data should be saved
    """

    # TODO Detect if the path already has a dataset in it; if it does, then don't generate images...

    # make blank numpy array
    image_matrix = np.full((sample_number, y_res, x_res, channels), v_min)

    if (v_min != 0 or v_max != 1) and save_to_folder == True:
        warnings.warr(f"v_min and v_max of 0-1 should be used when saving images!"
                      f"v_min and v_max of {v_min} and {v_max} found.")

    for i in range(sample_number):
        for j in range(label_count):
            # determine if a class should be used in an image



