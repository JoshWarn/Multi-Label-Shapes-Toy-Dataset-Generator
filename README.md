# Multi-Label Toy Dataset Generator

Multi-label neural networks can be challenging to make.
With many multilabel datasets being text, difficult to find, or having subjective, questionable labels, I created this generator to eliminate one point of error when making multi-label NNs; the dataset.

This generator makes it easy to confirm that a neural network is functioning as expected.
The recommended use case is to create an easy dataset that any neural network should do well on; however, if robustness is desired, the difficulty can be increased.
With this generator:
- images and labels are easily verifiable and boolean
- difficulty can be easily adjusted using generation parameters

Generation options include:
- number of samples to generate
- x_resolution, y_resolution, channels
- size of shapes (random between bounds or hard-set)
- frequency of shapes in image (random between bounds or hard-set)
- number of labels
- frequency of images to have a label

Additional options:
- Exporting the dataset into an image-folder

Current TODO:
- Make an option to export a pickle file
- Add more warnings, tracebacks, and comments
- Allow different classes to have different frequencies (currently only a single-hardset value)
- Add an option to get rid of no-label images (all zeros)

Examples:
256x256x3 5-labels dataset with all 5 classes (circles, lines, triangles, squares, pentagons):

![00020](https://github.com/JoshWarn/MultiLabelToyDatasetGenerator/assets/70070682/9b882357-44e8-4934-828c-c8d49bf0ae25)

Durations to generate and save datasets (may vary on differing hardware):
| Generation Parameters  | Generated (s)  | Generated + Saved (s) |
| :------------ |:---------------:| :-----:|
| 10000 256x256x3 3-label| 6.8 | 31 |
| 10000 256x256x3 5-label| 14.2| 39 |
