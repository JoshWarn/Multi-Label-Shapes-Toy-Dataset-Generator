import numpy as np
import skimage
import warnings
import traceback
from PIL import Image
import os
import time
import pickle
import timeit

def generate_multilabel_toy_dataset(sample_number=1000, x_res=256, y_res=256, channels=3, v_min=0, v_max=1,
                                    size=[10, 40], frequency=[2, 20], label_count=2, label_frequency=0.5,
                                    replace_no_label_images=False,
                                    path=None, export_type=None, verbose=True, all_channels_same_optim=True,
                                    progress_bar=False, random_seed=0):
    """
    Creates a dataset with basic shape perimeters; 1, 2, 3, 4... = circle, line, triangle, square etc...

    ~~ Image/Dataset options ~~
    sample_number   - number of samples to generate
    x_res, y_res    - dimensions of generated image
    channels        - number of image channels to use
    v_min, v_max    - minimum and maximum values used in channels (typically 0-1 for samples).
    size            - how large to draw the items in the image
    frequency       - how many items to draw on the image; can be set as a hard value or a range using [min, max]
    label_count     - number of classes to generate dataset with
    label_frequency - how frequently to have a label occur in an image; may be int or [max, min].
    replace_no_label_images - if images with no labels should be re-generated with labels; boolean.

    ~~ Saving options ~~
    path            - where to save the dataset to
    export_type  - if data should be saved in a certain format; None, "image_folder", "pickle".

    ~~ Misc ~~
    verbose         - if info about building the dataset should be printed; progress bar & time taken.
    all_channels_Same_optim - an optimization for if all channels are the same.
                      By default, this is enabled. Channels being different isn't supported (yet).
    progress_bar    - if a dataset-generation progress bar should be made. True/False
    TODO add more warnings and tracebacks.
    TODO add option to get rid of no-label images
    TODO add progress bar
    TODO rewrite comments, descriptions.
    TODO add channel specific classes(?)
    """

    valid_exports = ["image_folder", "pickle", None]
    if export_type not in valid_exports:
        warnings.warn(f"Export type {export_type} not supported! Valid list: {valid_exports}")
        raise BaseException("Unexpected export type!")

    if valid_exports is not None:  # Detects if folder exists or has files; makes folder if it doesn't exist.
        if not os.path.isdir(path):
            os.makedirs(path)
            os.makedirs(f"{path}\\Dataset")

    # Warnings and input checking
    if (v_min != 0 or v_max != 1) and export_type == "image_folder":
        warnings.warn(f"v_min and v_max of 0-1 should be used when saving images!"
                      f"v_min and v_max of {v_min} and {v_max} found.")

    if export_type is not None and path is None:
        warnings.warn(f"Exporting enabled ({export_type}), but no path provided! Path: {path}")

    # Warnings regarding frequency of items in image
    if type(frequency) is int:
        if frequency <= 0:
            warnings.warn(f"Frequency of {frequency} equal or below zero! Won't work.")
    elif type(frequency) is list:
        if len(frequency) != 2:
            warnings.warn(f"Too many frequency parameters! Expects a min and max bound but got {frequency}")
        if frequency[0] < 0:
            warnings.warn(f"Lower frequency bound of {frequency[0]} is less than zero! May break!")
        if frequency[0] == 0:
            warnings.warn(f"Lower frequency bound of {frequency[0]} is zero! May break (or result in mislables)!")
        if frequency[1] < frequency[0]:
            warnings.warn(f"Lower frequency bound {frequency[0]} greater than upper bound {frequency[1]}")
    else:
        warnings.warn(f"Frequency of {frequency} not supported!")

    # Warnings and processing of label frequency
    if type(label_frequency) is int or float:
        label_frequency_list = [[1 - label_frequency, label_frequency] for label in range(label_count)]
        if label_frequency >= 1 or label_frequency <= 0:
            warnings.warn(f"Label Frequency of {label_frequency} not supported!")
    else:
        # Make a list between min and max frequency; currently just linear
        label_frequency_temp = np.linspace(label_frequency[0], label_frequency[1], label_count)
        label_frequency_list = [[1-label_freq, label_freq] for label_freq in label_frequency_temp]
        if len(label_frequency) != 2:
            warnings.warn(f"Length of Label Frequency {len(label_frequency)} not supported!")

    # Warnings regarding size of items in image:
    if type(size) == int or type(size) == float:
        if size < 0 or size > x_res or size > y_res:
            warnings.warn(f"Size of {frequency} out of supported range!")
    elif type(size) is list:
        if len(size) != 2:
            warnings.warn(f"Too many size parameters! Expects a min and max bound but got {size}")
        if size[0] < 0:
            warnings.warn(f"Lower size bound of {size[0]} is less than zero! May break!")
        if size[0] == 0:
            warnings.warn(f"Lower size bound of {size[0]} is zero! May break (or result in mislables)!")
        if size[1] < size[0]:
            warnings.warn(f"Lower size bound {size[0]} greater than upper bound {size[1]}")
        if size[1] > x_res or size[1] > y_res:
            warnings.warn(f"Size of {size[1]} out of supported range!")

    # Value initialization
    start_time = time.time()
    np.random.seed(seed=random_seed)
    image_matrix = np.full((sample_number, channels, y_res, x_res), v_min, dtype=np.uint8)
    label_matrix = np.zeros((sample_number, label_count), dtype=np.uint8)

    for i in range(label_count):
        vertical_label_matrix = np.random.choice([0, 1], sample_number, p=label_frequency_list[i])
        label_matrix[:, i] = vertical_label_matrix

    for i in range(sample_number):
        for j in range(label_count):
            if label_matrix[i][j] == 1:
                image_matrix[i] = draw_shapes(image_matrix[i], j, size, frequency, v_max, x_res, y_res, all_channels_same_optim)

        if progress_bar and i%100 == 0:
            progressbar((i+1)/sample_number)

    image_matrix = np.transpose(image_matrix, (0, 2, 3, 1))     # Changing the order of dimensions
    if export_type == "image_folder":
        saved_img_matrix = image_matrix*255
        for i in range(len(image_matrix)):
            pil_img = Image.fromarray(saved_img_matrix[i].astype('uint8'))   # convert image to PIL image and save
            pil_img.save(f"{path}\\Dataset\\{i:05}.png", "PNG")

        with open(f"{path}\\labels.csv", "w") as f:  # Save label data
            for i in range(len(label_matrix)):
                txt_str = str(label_matrix[i]).replace("[", "").replace("]", "").replace(" ", ",")
                f.write(f"{txt_str}\n")
    elif export_type == "pickle":
        # TODO save the dataset as two files; an image array and a label array.
        with open(f"{path}\\Images.pkl", "wb") as file:
            pickle.dump(image_matrix, file)
        with open(f"{path}\\Labels.pkl", "wb") as file:
            pickle.dump(label_matrix, file)

    if verbose:
        print(f"Dataset of {sample_number} {x_res}x{y_res}x{channels} samples built in {time.time()-start_time:0f} seconds.")

    return image_matrix, label_matrix


def draw_shapes(image, label, size, frequency, v_max, x_res, y_res, all_channels_same):
    # Determines frequency of item in image; how many times to run the item loop.
    if type(frequency) is list:
        item_count = np.randint(frequency[0], frequency[1])         # Determines number of items to place in
    else:
        item_count = frequency

    for i in range(item_count):
        # Determining size of sample
        if type(size) is list:
            item_size = np.randint(size[0], size[1])
        else:
            item_size = size

        # Generate a circle within the range to use as bounds for the shape.
        ry = np.randint(0 + item_size, y_res - item_size - 1)
        cx = np.randint(0 + item_size, x_res - item_size - 1)

        # have to add 2 to the label to generate the correct number of points; generates 1 extra.
        # Generate an initial point at an angle of 0; maybe consider randomizing this later?
        # angle_list = (np.linspace(0, 2 * np.pi, label + 2)+random.random()*np.pi) % (2*np.pi)
        angle_list = ((np.linspace(0, 1, label + 2) + np.random.random_sample()) % 1) * 2*np.pi
        x_pos_list = (ry + np.cos(angle_list[0:-1]) * item_size).astype(int)
        y_pos_list = (cx + np.sin(angle_list[0:-1]) * item_size).astype(int)

        if all_channels_same:
            if label == 0:
                rr, cc = skimage.draw.circle_perimeter(ry, cx, int(item_size / 2))
                image[0][rr, cc] = v_max
            elif label == 1:
                for k in range(len(x_pos_list) - 1):
                    rr, cc = skimage.draw.line(y_pos_list[k], x_pos_list[k], y_pos_list[k + 1], x_pos_list[k + 1])
                    image[0][rr, cc] = v_max
            else:
                for k in range(len(x_pos_list)):
                    rr, cc = skimage.draw.line(y_pos_list[k - 1], x_pos_list[k - 1], y_pos_list[k], x_pos_list[k])
                    image[0][rr, cc] = v_max

            # sets all channels to the first channel
            for j in range(len(image)-1):
                image[j + 1] = image[0]

        else:
            for j in range(len(image)):
                if label == 0:  # draw circle
                    rr, cc = skimage.draw.circle_perimeter(ry, cx, int(item_size/2))
                    image[j][rr, cc] = v_max

                # else, generate the line(s)from the circle's perimeter.
                # minor speed improvement to stop overdrawing lines twice for lines:
                elif label == 1:
                    for k in range(len(x_pos_list)-1):
                        rr, cc = skimage.draw.line(y_pos_list[k], x_pos_list[k], y_pos_list[k+1], x_pos_list[k+1])
                        image[j][rr, cc] = v_max
                else:
                    for k in range(len(x_pos_list)):
                        rr, cc = skimage.draw.line(y_pos_list[k-1], x_pos_list[k-1], y_pos_list[k], x_pos_list[k])
                        image[j][rr, cc] = v_max
    return image


def progressbar(percent):
    # This is just a placeholder!


# Example usage:
images = generate_multilabel_toy_dataset(10000, label_count=3, frequency=[2, 20], path="Dataset", export_type='pickle')


# print(timeit.repeat("generate_multilabel_toy_dataset(10000, label_count=5, frequency=[2, 20], path='Dataset', export_type=Non)",
#                     "from __main__ import generate_multilabel_toy_dataset", repeat=10, number=1))
"""
Time statistics:
8/29/2023 6:15 PM
10k-3l: 6.9095, 6.965, 7.092, 7.134, 6.986
10k-5l: 14.293532099982258, 14.326442399993539, 14.194133099983446, 14.242385699995793, 14.304134599980898
"""