# DLAD
Code and models for the preliminary screening of pulmonary diseases from X-ray images based on an improved variational autoencoder network.

# Run:

The /output subfolder contains the anomaly scores from the corresponding model's test results and the encoded features extracted from each test sample.

/split_dataset records the dataset partitioning situation (only includes the remaining samples after the difficult case selection in the first research task).

`my_dataset.py` for dataset reading.

`network.py` contains the model construction code.

`network_1.py` uses ModuleList to construct the model (more convenient, but prone to errors during training).

train.py for model training. (`train_1.py` and `train_2.py` represent two alternative training methods).
# Test:

`test.py` for testing code.


# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* TensorBoardX == 2.6.22
* Python == 3.10.14
* numpy == 1.26.4
* opencv-python == 4.10.0.84
* scipy == 1.14.0