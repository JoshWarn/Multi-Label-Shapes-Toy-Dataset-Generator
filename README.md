# Multi-Label "Shapes" Toy Dataset Generator

Multi-label neural networks can be challenging to make.
With many multilabel datasets being text, difficult to find, or having subjective questionable labels, I created this generator to eliminate one point of error when making multi-label NNs; the dataset.

This generator makes it easy to confirm that a neural network is functioning as expected.

The recommended use case is to create an easy dataset that any neural network should do well on as a basic verification that everything is working as expected; however, if robustness is desired, the difficulty can be increased.

### With this generator:
- Images and labels are easily verifiable.
- Dataset difficulty can be easily adjusted using generation parameters.

### Base Recommended Parameters:

>generate_multilabel_toy_dataset(sample_number=10000,
                                label_count=3,
                                x_res=256, y_res=256, channels=3,
                                v_min=0, v_max=1,
                                size=[10, 40],
                                frequency=[2, 20],
                                label_count=3,
                                label_frequency=0.5,
                                opacity=1,
                                path="",
                                export_folder="ShapesDataset"
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
- Opacity of generated shapes
### Additional options:
- Exporting the dataset into an image-folder/pickle file
- Setting a seed for random numbers
- If messages should be shown (progress bar, comments)

### Features TODO:
- ~~Implement an opacity option.~~ (Added 9/4, Done 9/6/23)
- ~~Detect files in dataset folder and replace them if generating a new dataset.~~ (Added 8/30, Done 9/2/23)
- Add more warnings, tracebacks, and comments. (Added 8/27, Ongoing)
- ~~Add an option to get rid of no-label images (all zeros).~~ (Added 8/27, Removed 8/30/23)
- ~~Add a progress bar.~~ (Added 8/27, Done 8/30/23)
- ~~Add a set seed for random numbers.~~ (Added 8/27, Done 8/30/23)
- ~~Add channel-specific classes/shapes.~~ (Added 8/27, Done 8/30/23)
- ~~Make an option to export a pickle file.~~ (Added 8/27, Done 8/29/23)
- ~~Allow different classes to have different frequencies (currently only a single-hard-set value).~~ (Added 8/27, Done 8/29/23)

### Examples:
3-labels: (circles, lines, triangles)

5-labels: (circles, lines, triangles, squares, pentagons)
|256x256x3 5-labels w/ all 5 classes:| 256x256x3 5-labels w/ 5 classes & random_channel_classes:|
|:---------------:|:---------------:|
| ![00020](https://github.com/JoshWarn/MultiLabelToyDatasetGenerator/assets/70070682/9b882357-44e8-4934-828c-c8d49bf0ae25) |![00086](https://github.com/JoshWarn/Multi-Label-Shapes-Toy-Dataset-Generator/assets/70070682/f01bf01a-7ef5-49fc-adf1-30cb6fd05fad)|



### Durations to generate and save datasets (may vary on differing hardware):
| Generation Parameters  | Generated (s) | Generated + Pickle (s) | Generated + Image_Folder (s) |
| :------------ |:---------------:|:-----:|:-----:|
| 10,000 256x256x3 3-label Default | 5.0 | 16.5 | 30.4 |
| 10,000 256x256x3 5-label Default | 10.8 | 22.2 | 40.4 |
