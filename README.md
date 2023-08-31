# Multi-Label Toy Dataset Generator

Multi-label neural networks can be challenging to make.
With many multilabel datasets being text, difficult to find, or having subjective questionable labels, I created this generator to eliminate one point of error when making multi-label NNs; the dataset.

This generator makes it easy to confirm that a neural network is functioning as expected.

The recommended use case is to create an easy dataset that any neural network should do well on as a basic verification that everything is working as expected; however, if robustness is desired, the difficulty can be increased.

### With this generator:
- Images and labels are easily verifiable.
- Dataset difficulty can be easily adjusted using generation parameters.

### Base Recommended Parameters:

generate_multilabel_toy_dataset(sample_number=10000,
                                label_count=3,
                                x_res=256, y_res=256, channels=3,
                                v_min=0, v_max=1,
                                size=[10, 40],
                                frequency=[2, 20],
                                label_count=3,
                                label_frequency=0.5,
                                path='Dataset',
                                export_type="image_folder",
                                verbose=True,
                                random_seed=0,
                                random_channel_classes=False)

### Generation options include:
- Number of samples
- X_res, R_res, Channels
- Number of labels
- Shape Sizes (random between bounds or hard-set)
- Frequency of shapes in image (random between bounds or hard-set)
- Frequency of images to have a label (linear space between bounds or hard-set)
- If shapes should be generated to random channels

### Additional options:
- Exporting the dataset into an image-folder/pickle file
- Setting a seed for random numbers
- If messages should be shown (progress bar, comments)

### Features TODO:
- Detect files in dataset folder and ask user if dataset generation should continue, replacing them. (Added 8/30)
- Add more warnings, tracebacks, and comments. (Added 8/27)
- ~~Add an option to get rid of no-label images (all zeros).~~ (Added 8/27, Removed 8/30/23)
- ~~Add a progress bar.~~ (Added 8/27, Done 8/30/23)
- ~~Add a set seed for random numbers.~~ (Added 8/27, Done 8/30/23)
- ~~Add channel-specific classes/shapes.~~ (Added 8/27, Done 8/30/23)
- ~~Make an option to export a pickle file.~~ (Added 8/27, Done 8/29/23)
- ~~Allow different classes to have different frequencies (currently only a single-hardset value).~~ (Added 8/27, Done 8/29/23)

### Examples:
256x256x3 5-labels dataset with all 5 classes (circles, lines, triangles, squares, pentagons):

![00020](https://github.com/JoshWarn/MultiLabelToyDatasetGenerator/assets/70070682/9b882357-44e8-4934-828c-c8d49bf0ae25)

### Durations to generate and save datasets (may vary on differing hardware):
| Generation Parameters  | Generated (s) | Generated + Image_Folder (s) | Generated + Pickle (s) |
| :------------ |:---------------:|:-----:|:-----:|
| 10,000 256x256x3 3-label| 6.8 | 31 | XX |
| 10,000 256x256x3 5-label| 14.2| 39 | XX |
