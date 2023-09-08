import os
import time
import pickle
import warnings
import numpy as np
from PIL import Image
import skimage

# Copyright (c) 2023 Joshua Warnasch (Light MIT License)

# Permission is freely granted to persons obtaining a copy of this software and
# documentation files (the "Software"), to deal in the Software without limits
# such as rights to use/copy/modify/merge/publish/distribute/sublicense and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# These notices shall be carried in all copies/branches of the Software.

# SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY IMPLIED/EXPRESSED WARRANTY, INCLUDING
# BUT NOT LIMITED TO MERCHANTABILITY/FITNESS/NONINFRINGEMENT. HOLDERS SHALL NOT
# BE LIABLE FOR CLAIMS/DAMAGES/LIABILITY, WHETHER IN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING DUE TO ANY DEALINGS IN THE SOFTWARE.


def generate_multilabel_toy_dataset(sample_number=1000, x_res=256, y_res=256, channels=3, v_min=0, v_max=1,
                                    size=(10, 40), frequency=(2, 20), label_count=3, label_frequency=0.5, opacity=1,
                                    path="", export_folder="ShapesDataset", export_type=None, verbose=True,
                                    random_seed=0, random_channel_classes=False):
    """
    Creates a dataset with basic shape perimeters; 1, 2, 3, 4... = circle, line, triangle, square etc...
    ~~ Image/Dataset options ~~
    sample_number   - int; number of samples to generate
    x_res, y_res    - int(s); dimensions of generated image
    channels        - int; number of image channels to use
    v_min, v_max    - int(s); minimum and maximum values used in channels (typically 0-1 for samples).
    size            - int/[int, int]; how large to draw the items in the image
    frequency       - int/[int, int]; how many items to draw on image; may be set single value or range using [min, max]
    label_count     - int; number of classes to generate dataset with
    label_frequency - int/[int, int]; how frequently to have a label occur in an image; may be int or [max, min]
    opacity         - float; how transparent a generated shape is; 0 = transparent, 1 is opaque.

    ~~ Saving options ~~
    path            - str; where to save the dataset to
    export_folder   - str; name of the dataset folder
    export_type     - str; if data should be saved in a certain format; None, "image_folder", "pickle".

    ~~ Misc ~~
    verbose         - bool; if info about building the dataset should be printed; progress bar & time taken.
    random_seed     - int; sets random state; default 0.
    random_channel_classes - bool; allows channels to show up on random channels. Default disabled.

    TODO: Sometimes label file is wrong?!? Fix.
    """

    # ~~ Input sanitization and verification ~~
    check_input_validity(sample_number, x_res, y_res, channels, v_min, v_max, size, frequency,
                         label_count, label_frequency, opacity, path, export_folder, export_type,
                         verbose, random_seed, random_channel_classes)

    # ~~ Value initialization and pre-processing ~~
    start_time = time.time()
    rng = np.random.default_rng(seed=random_seed)
    np.random.seed(seed=random_seed)
    image_matrix = np.full((sample_number, channels, y_res, x_res), v_min, dtype=np.float32)
    label_matrix = np.zeros((sample_number, label_count), dtype=np.bool_)

    # If path provided use it; else get local project path.
    path = path if path else os.path.abspath(os.getcwd())

    # Detects if folder exists or has files; makes folder if it doesn't exist.
    manage_export_path(export_type, path, export_folder, verbose)

    # Created label probabilities for each label
    if type(label_frequency) is int or float:
        label_frequency_list = [[1 - label_frequency, label_frequency] for _ in range(label_count)]
    else:
        # Make a list between min and max frequency; currently just linear
        label_frequency_temp = np.linspace(label_frequency[0], label_frequency[1], label_count)
        label_frequency_list = [[1-label_freq, label_freq] for label_freq in label_frequency_temp]

    # Initializes values into a sample_num x label_count matrix with a given frequency
    for i in range(label_count):
        vertical_label_matrix = np.random.choice([0, 1], sample_number, p=label_frequency_list[i])
        label_matrix[:, i] = vertical_label_matrix

    # ~~ Main Generation Loop ~~
    if verbose:
        print("Generating samples...")

    for i in range(sample_number):
        # Updating the progress bar after each 100 samples
        if verbose and i % 100 == 0:
            progressbar(i/sample_number)

        # For each label, if True draw shapes
        for j in range(label_count):
            if label_matrix[i][j]:
                image_matrix[i] = draw_shapes(image_matrix[i], j, size, frequency, opacity*v_max, x_res, y_res,
                                              random_channel_classes, rng)

    # Setting progress bar to completed
    if verbose:
        progressbar(1.)

    # ~~ Postprocessing ~~
    # Clipping values to v_max: This is due to a poor solution when combining channels for different shapes.
    image_matrix = np.clip(image_matrix, v_min, v_max)

    # Changing the order of dimensions in image matrix to make saving and opening easy.
    image_matrix = np.transpose(image_matrix, (0, 2, 3, 1))

    # ~~ Saving ~~
    if export_type is not None:
        if verbose:
            print(f"Saving dataset to {export_type}...")

        # Exporting images
        export_images(image_matrix, label_matrix, path, export_folder, export_type, verbose)

    if verbose:
        print(f"Dataset of {sample_number} {x_res}x{y_res}x{channels} samples built in {time.time()-start_time:0f} s.")
    return image_matrix, label_matrix


def check_input_validity(sample_number, x_res, y_res, channels, v_min, v_max, size, frequency,
                         label_count, label_frequency, opacity, path, export_folder, export_type,
                         verbose, random_seed, random_channel_classes):
    """
    Wrapper function to checks validity of all inputs; broken into a separate function for code-cleanliness
    """
    input_validity(var_val=sample_number, var_name="Sample-Count", var_dtypes=[int], var_min=1, var_max=1E9,
                   series_len=1, series_trend=None, actions=["raise", "raise", "warn", "raise", "raise"])
    input_validity(var_val=x_res, var_name="X-Resolution", var_dtypes=[int], var_min=8, var_max=2 ** 10,
                   series_len=1, series_trend=None, actions=["raise", "raise", "warn", "raise", "raise"])
    input_validity(var_val=y_res, var_name="Y-Resolution", var_dtypes=[int], var_min=8, var_max=2 ** 10,
                   series_len=1, series_trend=None, actions=["raise", "raise", "warn", "raise", "raise"])
    input_validity(var_val=channels, var_name="Channel-Count", var_dtypes=[int], var_min=1, var_max=2 ** 8,
                   series_len=1, series_trend=None, actions=["raise", "raise", "warn", "raise", "raise"])
    input_validity(var_val=v_min, var_name="Min-Img-Value", var_dtypes=[int, float], var_min=-float("inf"),
                   var_max=float("inf"), series_len=1, series_trend=None,
                   actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=v_max, var_name="Max-Img-Value", var_dtypes=[int, float], var_min=-float("inf"),
                   var_max=float("inf"), series_len=1, series_trend=None,
                   actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=size, var_name="Item-Size", var_dtypes=[int, float, list, tuple],
                   var_min=4, var_max=min(x_res, y_res), series_len=2, series_trend="increasing",
                   actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=frequency, var_name="Item-Frequency", var_dtypes=[int, list, tuple],
                   var_min=1, var_max=float("inf"), series_len=2, series_trend=None,
                   actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=label_count, var_name="Label-Count", var_dtypes=[int], var_min=1, var_max=float("inf"),
                   series_len=1, series_trend=None, actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=label_frequency, var_name="Label-Frequency", var_dtypes=[float, list, tuple],
                   var_min=0, var_max=1, series_len=2, series_trend="increasing",
                   actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=opacity, var_name="Opacity", var_dtypes=[int, float],
                   var_min=0+1E-10, var_max=1, series_len=1, series_trend=None,
                   actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=path, var_name="Path", var_dtypes=[str], var_min="", var_max="",
                   series_len=1, series_trend="", actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=export_folder, var_name="Export-Folder", var_dtypes=[str], var_min="", var_max="",
                   series_len=1, series_trend="", actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=export_type, var_name="Export-Type", var_dtypes=[str, type(None)], var_min="", var_max="",
                   series_len=1, series_trend="", actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=verbose, var_name="Verbosity", var_dtypes=[bool], var_min="", var_max="",
                   series_len=1, series_trend="", actions=["warning", "raise", "raise", "raise", "raise"])
    input_validity(var_val=random_seed, var_name="Random-Seed", var_dtypes=[int],
                   var_min=-float("inf"), var_max=float("inf"), series_len=0, series_trend="",
                   actions=["raise", "raise", "raise", "raise", "raise"])
    input_validity(var_val=random_channel_classes, var_name="Random-Channel-Classes", var_dtypes=[bool],
                   var_min="", var_max="", series_len=1, series_trend="",
                   actions=["raise", "raise", "raise", "raise", "raise"])

    # Additional cleaning/verification for non-standard variables
    # Make sure valid_export is in list
    valid_exports = ["image_folder", "pickle", None]
    if export_type not in valid_exports:
        warnings.warn(f"Export type {export_type} not supported! Valid list: {valid_exports}")
        raise Exception("Unexpected export type!")

    # Raising a warning if value bounds aren't [0, 1] when exporting to image_folder:
    if (v_min != 0 or v_max != 1) and export_type == "image_folder":
        warnings.warn(f"v_min and v_max of 0-1 should be used when saving images!"
                      f"v_min and v_max of {v_min} and {v_max} found.")


def input_validity(var_val, var_name, var_dtypes, var_min=None, var_max=None,
                   series_len=1, series_trend=None, actions=("raise", "raise", "raise", "raise", "raise")):
    """
    Sends messages to console for variable validating.
    Exceptions sourced from: https://docs.python.org/3/library/exceptions.html#exception-hierarchy

    :param var_val:             Value entered for variable
    :param var_name:            Name to print when referring to variable
    :param var_dtypes:  Accepted dtypes of variable
    :param var_min:             Minimum accepted value of variable
    :param var_max:             Maximum accepted value of variable
    :param series_len:          Expected variable length (only used if var_accepted_types has "list").
    :param series_trend:        Trend of values in list; "None" (bypassing), "increasing", "decreasing".
                                Used for optional [Min, Max] entries.
    :param actions:             What to do if triggered; "warn" or "raise" accepted.
                                Order is [var_val_type, min_val, max_val, series_len, series_trend]
    :return:                    None.
    """

    # Gathering information about variable
    var_val_type = type(var_val)

    # Error messages:
    dtype_error_msg = f"Expected {var_name} dtype in {var_dtypes}, but {var_val_type} was found."
    min_val_error_msg = f"Value of {var_val} in {var_name} is less than {var_min}."
    max_val_error_msg = f"Value of {var_val} in {var_name} is more than {var_max}."
    series_len_error_msg = f"Variable length of {var_val} in {var_name} isn't equal to {series_len}."
    series_trend_error_msg = f"Series trend of {var_name} expected to be {series_trend}, but was {var_val}."
    series_trend_invalid_msg = (f"Series trend of {var_name} expected to be in [None, '', 'increasing', 'decreasing'],"
                                f"but was {series_trend}")

    # Checking variable type
    if var_val_type not in var_dtypes:
        if actions[0] == "warn":
            warnings.warn(dtype_error_msg)
        elif actions[0] == "raise":
            raise TypeError(dtype_error_msg)

    # Checking variable min/max
    # If variable has multiple values, check each. Otherwise, check single value
    # This may not catch if numpy inputs are passed; perhaps revise later.
    if var_val_type in [list, tuple]:
        for val in var_val:
            if type(val) in [float, int]:
                if var_min is not None:
                    if val < var_min:
                        if actions[1] == "warn":
                            warnings.warn(min_val_error_msg)
                        elif actions[1] == "raise":
                            raise ValueError(min_val_error_msg)
                if var_max is not None:
                    if val > var_max:
                        if actions[2] == "warn":
                            warnings.warn(max_val_error_msg)
                        elif actions[2] == "raise":
                            raise ValueError(max_val_error_msg)
    else:
        if var_val_type in [float, int]:
            if var_min is not None:
                if var_val < var_min:
                    if actions[1] == "warn":
                        warnings.warn(min_val_error_msg)
                    elif actions[1] == "raise":
                        raise Exception(min_val_error_msg)
            if var_max is not None:
                if var_val > var_max:
                    if actions[2] == "warn":
                        warnings.warn(max_val_error_msg)
                    elif actions[2] == "raise":
                        raise Exception(max_val_error_msg)

    # If series, ensuring length is valid
    if var_val_type in [list, tuple] and series_len != "":  # Don't know if this second part is required.
        if len(var_val) != series_len:
            if actions[3] == "warn":
                warnings.warn(series_len_error_msg)
            elif actions[3] == "raise":
                raise TypeError(series_len_error_msg)

    # Ensure series_trend is valid
    if series_trend not in [None, "", "increasing", "decreasing"]:
        if actions[4] == "warn":
            warnings.warn(series_trend_invalid_msg)
        elif actions[4] == "raise":
            raise Exception(series_trend_invalid_msg)

    # If series, make sure trend is correct.
    # Make sure each value is larger/smaller than the previous
    if var_val_type in ["list", "tuple"] and series_trend in ["increasing", "decreasing"]:
        if series_trend == "increasing":
            for i in range(len(var_val)-1):
                # If next variable value isn't larger:
                if var_val[i] >= var_val[i+1]:
                    if actions[4] == "warn":
                        warnings.warn(series_trend_error_msg)
                    elif actions[4] == "raise":
                        raise Exception(series_trend_error_msg)
        elif series_trend == "decreasing":
            for i in range(len(var_val)-1):
                # If next variable value isn't smaller:
                if var_val[i] <= var_val[i+1]:
                    if actions[4] == "warn":
                        warnings.warn(series_trend_error_msg)
                    elif actions[4] == "raise":
                        raise Exception(series_trend_error_msg)


def draw_shapes(image, label, size, frequency, opacity, x_res, y_res, random_channel_classes, rng):
    # Determines frequency of item in image; how many times to run the item loop.
    if type(frequency) in [list, tuple]:
        item_count = rng.integers(frequency[0], frequency[1])
    else:
        item_count = frequency

    # Selecting what channel(s) to put shapes on:
    if random_channel_classes:
        # Using randint to make a boolean "channels-used" array.
        # Ensures that at least 1 channel has the shapes.
        num_used_channels = rng.integers(1, len(image))
        channels_used = np.array([True] * num_used_channels + [False]*(len(image)-num_used_channels), dtype=bool)

        # Shuffle to use random channels
        np.random.shuffle(channels_used)
    else:
        # If not random, select all channels.
        channels_used = np.ones(len(image), dtype=bool)

    # Determining size of sample diameters
    if type(size) in [list, tuple]:
        item_size_list = rng.integers(size[0], size[1], item_count)
    else:
        item_size_list = np.full(item_count, size)

    # Make a zeros-array of the image-shape.
    shape_layer = np.zeros((y_res, x_res), dtype=float)

    # Generating random centers within ranges to use as bounds for the shapes.
    ry_l = ((rng.random(item_count) * (y_res - (item_size_list+2))) + item_size_list/2 + 1).astype(int)
    cx_l = ((rng.random(item_count) * (x_res - (item_size_list+2))) + item_size_list/2 + 1).astype(int)
    if label == 0:
        for i in range(item_count):
            rr, cc = skimage.draw.circle_perimeter(ry_l[i], cx_l[i], int(item_size_list[i] / 2))
            shape_layer[rr, cc] += opacity
    else:
        # Label + 2 yields correct point-spacing in circle; last point not used.
        linear_space = np.tile(np.linspace(0, 1, label + 2)[0:-1], (item_count, 1))
        random_angle_offset = np.reshape(rng.random(size=item_count), (item_count, 1))
        angle_matrix = (linear_space + random_angle_offset % 1) * 2*np.pi

        item_size_list = np.reshape(item_size_list, (item_count, 1))

        ry_l = np.reshape(ry_l, (item_count, 1))
        cx_l = np.reshape(cx_l, (item_count, 1))

        # Divided Item_size by 2 to properly use radii instead of diameter.
        x_pos_matrix = (cx_l + np.cos(angle_matrix) * item_size_list/2).astype(int)
        y_pos_matrix = (ry_l + np.sin(angle_matrix) * item_size_list/2).astype(int)

        if label == 1:
            for i in range(item_count):
                for k in range(len(x_pos_matrix[i]) - 1):
                    rr, cc = skimage.draw.line(y_pos_matrix[i][k], x_pos_matrix[i][k], y_pos_matrix[i][k + 1], x_pos_matrix[i][k + 1])
                    shape_layer[rr, cc] += opacity
        else:
            for i in range(item_count):
                for k in range(len(x_pos_matrix[i])):
                    rr, cc = skimage.draw.line(y_pos_matrix[i][k - 1], x_pos_matrix[i][k - 1], y_pos_matrix[i][k], x_pos_matrix[i][k])
                    shape_layer[rr, cc] += opacity

    # sets all other channels to the first-used-channel if used.
    for j in range(len(image)):
        if channels_used[j]:
            # Originally deemed sketchy, now needed due to use of opacity.
            image[j] += shape_layer

    return image


def progressbar(percent, bar_len=50):
    """
    Generates and prints a simple progress bar.
    :param percent: float between - and 1; percent complete.
    :param bar_len: total length of bar (int)
    :return: None
    """
    # Input sanitization/verification
    if not 0 <= percent <= 1 or type(percent) not in [float, int]:
        print(f"Percentage value of Value:{percent} Type:{type(percent)} not supported!")

    if bar_len <= 1 or type(bar_len) is not int:
        print(f"Bar_len value of Value:{bar_len} Type:{type(bar_len)} not supported!")

    # Calculations:
    bar = int(percent*bar_len)
    # Typically end="", but issues w/ IDEs not supporting return carriage "\r" resulting in slowdowns (ex. IDLE).
    if percent == 1:
        print(f"\rProgress: |{'█' * bar + '─' * (bar_len - bar)}| {100 * percent:f}%", end="\n")
    else:
        print(f"\rProgress: |{'█' * bar + '─' * (bar_len - bar)}| {100 * percent:f}%", end="")


def export_images(image_matrix, label_matrix, path, export_folder, export_type, verbose):
    if export_type == "image_folder":
        # Multiplying by 255 to get [0, 255] values for PIL
        saved_img_matrix = image_matrix*255
        # Gets the digit count of the dataset to use in file-naming.
        img_number_length = f"{len(str(len(image_matrix))):02}"
        for i in range(len(image_matrix)):
            # If verbose, create progress bar.
            if verbose and i % 100 == 0:
                progressbar(i / len(saved_img_matrix))
            # convert image to PIL image and saving the image
            pil_img = Image.fromarray(saved_img_matrix[i].astype('uint8'))
            # save image
            pil_img.save(f"{path}\\{export_folder}\\{i:{img_number_length}}.png", "PNG")

        # Save label data
        with open(f"{path}\\{export_folder}\\labels.csv", "w") as f:
            for i in range(len(label_matrix)):
                # Might be a better way to do this than to use all the replace commands...
                # Issues with numpy boolean array; there's an extra white-space before "True" to pad with "False".
                txt_str = str(label_matrix[i]).replace("[", "").replace("]", "").replace(" True", "True").replace(" ", ", ")
                f.write(f"{i:{img_number_length}}.png, {str(txt_str)}\n")
        if verbose:
            progressbar(1.)

    elif export_type == "pickle":
        if verbose:
            progressbar(0)

        # Save pickle files directly to folder.
        with open(f"{path}\\{export_folder}\\Images.pkl", "wb") as file:
            pickle.dump(image_matrix, file)
        with open(f"{path}\\{export_folder}\\Labels.pkl", "wb") as file:
            pickle.dump(label_matrix, file)

        if verbose:
            progressbar(1.)


def manage_export_path(export_type, path, export_folder, verbose):
    if export_type is not None:
        # base directory doesn't exist, send a warning, but create it anyway.
        if os.path.isdir(path) is False:
            os.makedirs(path)
            if verbose:
                warnings.warn("Path provided may be one level too deep!"
                              "This may result in the Dataset folder being in a sub-folder of the project.")

        # If Datafolder doesn't exist; create it.
        if os.path.isdir(f"{path}\\{export_folder}") is False:
            os.makedirs(f"{path}\\{export_folder}")
            if verbose:
                print("Creating Dataset folder...")

        # Else, check and delete files within folder; otherwise, create the dataset folder.
        else:
            if verbose:
                print("Main Export Folder already exists!")
            # Gets all files and folders within the directory
            files = [[dirpath, dirname, filename] for (dirpath, dirname, filename) in os.walk(f"{path}\\{export_folder}")]

            # Determines if there are files within the export folder.
            files_found = False
            for location in files:
                if len(location[2]) != 0:
                    files_found = True
            if files_found and verbose:
                warnings.warn(f"Files already found in or under folder: {path}\\{export_folder}. "
                              f"Writing over!")

            # Deleting all files in the dataset folder
            for location in files:
                if len(location[2]) > 0:
                    for file in location[2]:
                        os.remove(f"{location[0]}\\{file}")

            # No sub-folders should exist in Dataset folder unless user-made.
            # Can't delete them w/o admin privileges; instead just warn user.
            if verbose:
                for location in files:
                    if len(location[1]) > 0:
                        for folder in location[1]:
                            warnings.warn(f"Sub-folder(s) '{folder}' within dataset path: {path}\\{export_folder}.")
