import numpy as np
import skimage
import warnings
import random
import traceback
import math
from PIL import Image
import os


def generate_multilabel_toy_dataset(sample_number=1000, x_res=256, y_res=256, channels=3, v_min=0, v_max=1,
                                    size=[10, 40], frequency=[2, 20], label_count=2, label_frequency=0.5,
                                    path=None, save_to_folder=False):
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
    label_frequency - how frequently to have a label occur in an image

    ~~ Saving options ~~
    path            - where to save the dataset to
    save_to_folder  - if data should be saved
    """

    if save_to_folder is True:  # Detects if folder exists or has files; makes folder if it doesn't exist.
        if os.path.isdir(path):
            for (dirpath, dirnames, filenames) in os.walk(path):
                if len(filenames) != 0:
                    raise Exception("Data folder already populated with files!")
        else:
            os.makedirs(path)

    # Warnings and input checking
    if (v_min != 0 or v_max != 1) and save_to_folder is True:
        warnings.warn(f"v_min and v_max of 0-1 should be used when saving images!"
                      f"v_min and v_max of {v_min} and {v_max} found.")

    if save_to_folder is True and path is None:
        warnings.warn(f"Save_to_folder set to true, but no path provided! Path: {path}")

    # Warnings regarding frequency of items in image
    if type(frequency) is int:
        # TODO add raise and exceptions
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

    image_matrix = np.full((sample_number, channels, y_res, x_res), v_min, dtype=np.uint8)
    label_matrix = np.zeros((sample_number, label_count), dtype=np.uint8)

    for i in range(sample_number):
        for j in range(label_count):
            if random.random() > 1 - label_frequency:   # determine if a class should be used in an image
                label_matrix[i][j] = 1                 # set label to true
                image_matrix[i] = draw_shapes(image_matrix[i], j, size, frequency, v_max, x_res, y_res)

    image_matrix = np.transpose(image_matrix, (0, 2, 3, 1))     # Changing the order of dimensions
    if save_to_folder is True:
        saved_img_matrix = image_matrix*255
        for i in range(len(image_matrix)):
            pil_img = Image.fromarray(saved_img_matrix[i].astype('uint8'))   # convert image to PIL image and save
            pil_img.save(f"{path}/{i:05}.png", "PNG")

        with open(f"{path}\\labels.csv", "w") as f:  # Save label data
            for i in range(len(label_matrix)):
                txt_str = str(label_matrix[i]).replace("[", "").replace("]", "").replace(" ", ",")
                f.write(f"{txt_str}\n")

    return image_matrix, label_matrix


def draw_shapes(image, label, size, frequency, v_max, x_res, y_res):
    # Determines frequency of item in image; how many times to run the item loop.
    if type(frequency) is list:
        item_count = random.randint(frequency[0], frequency[1])   # Determines number of items to place in
    else:
        item_count = frequency

    for i in range(item_count):
        # Determining size of sample
        if type(size) is list:
            item_size = random.randint(size[0], size[1])
        else:
            item_size = size

        if label == 1:  # If label is a line, determine coordinates of line
            x1, y1 = -1, -1
            while x1 < 0 or x1 >= x_res or y1 < 0 or y1 >= y_res:   # try until all points are valid; slow but easy.
                # Select a random starting point
                x0, y0 = random.randint(0, x_res-1), random.randint(0, y_res-1)   # Select a random starting point
                ang = random.uniform(0, math.pi*2)                                  # Select a random angle
                x1, y1 = int(x0 + math.cos(ang)*item_size),  int(y0 + math.sin(ang)*item_size)  # calc endpoints
        else:   # Is a circle or n-gon; determine center position
            ry = random.randint(0 + item_size, y_res - item_size - 1)
            cx = random.randint(0 + item_size, x_res - item_size - 1)

        for j in range(len(image)):
            if label == 0:  # draw circle
                rr, cc = skimage.draw.circle_perimeter(ry, cx, int(item_size/2))
                image[j][rr, cc] = v_max

            if label == 1:  # draw line
                rr, cc = skimage.draw.line(y0, x0, y1, x1)
                image[j][rr, cc] = v_max

            if label >= 2:  # draw normal n-gon
                # TODO NOT implemented!
                print("Polygon drawing not implemented!")
    return image


generate_multilabel_toy_dataset(100, path="DataFolder", save_to_folder=True)
