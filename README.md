# CMU-11685-HW2P2

Fall 2021 Introduction to Deep Learning - Homework 2 Part 2 (face classification, face verification)

Author: Ryan Nelson

```text
               _---_
             /       \
            |         |
    _--_    |         |    _--_
   /__  \   \  0   0  /   /  __\
      \  \   \       /   /  /
       \  -__-       -__-  /
   |\   \    __     __    /   /|
   | \___----         ----___/ |
   \                           /
    --___--/    / \    \--___--
          /    /   \    \
    --___-    /     \    -___--
    \_    __-         -__    _/
      ----               ----
```

## Introduction

`octopus` is a python module that standardizes the execution of deep learning pipelines using `pytorch`, `wandb`,
and `kaggle`. Module behavior is controlled using a configuration file. The version of `octopus` in this repository has
been updated from the previous version to enable use with image data and CNNs.

## Features

- Allows user to control all module behavior through a single configuration file (see Configuration File Options)
- Can process multiple configuration files in a single execution to facilitate hyperparameter tuning
- (Optionally) downloads and unzips data from kaggle (or is able to use data already available)
- Uses `wandb` to track model statistics, save logs, and calculate early stopping criteria
- Generates a log file for each run
- Automatically saves checkpoints of the model and statistics after each epoch
- Allows user to start running a model from a previously saved checkpoint with minimal effort
- Automatically saves test output after each epoch
- Allows user to customize datasets, evaluation, and test output formatting depending on the data
- Predefined models can be customized via configuration file to facilitate hyperparameter tuning

## Requirements

- `wandb` account: Running this code requires a `wandb` account.
- `kaggle.json` file: If you choose to download data from kaggle, you'll need access to an API key file for your kaggle
  account.
- For image data (`data_type=image`), expects training and validation datasets to be organized for use
  with `ImageFolder`.

## To Run the Code

1. The code for this model consists of the following components:
    - python module `octopus`
    - python module `customized`
    - bash script `mount_drive.sh`

2. Activate an environment that contains torch, torchvision, pandas, numpy, and PIL. I used `pytorch_latest_p37`
   environment as part of the Deep Learning AMI for AWS EC2 instances.

3. Define each configuration file as desired. Configuration file options are listed below.

4. If your instance has a mountable storage drive `/dev/nvme1n1`, execute the bash script to mount the drive and change
   permissions to allow the user to write to directories inside the drive.

```bash
$ bash mount_drive.sh 
```

5. Execute the code using the following command from the shell to run `octopus` for each configuration file in the
   configuration directory. Files will be processed in alphabetical order.

```bash
$ python run_octopus.py path/to/config_directory
```

## Configuration File Options

Configurations must be parsable by `configparser`. To use the verification feature in this repository, set the
competition to the verification one.

```text
[DEFAULT]
run_name = Resnet34-Run-053  # sets the name of the run in wandb, log, checkpoints, and output files 

[debug]
debug_path = /home/ubuntu/  # where log file will be saved

[kaggle]
download_from_kaggle = True                            # whether to download data from kaggle
kaggle_dir = /home/ubuntu/.kaggle                      # fully qualified location of kaggle directory
content_dir = /home/ubuntu/content/                    # fully qualified location where kaggle data will be downloaded
token_file = /home/ubuntu/kaggle.json                  # where kaggle api file should be placed
competition = idl-fall21-hw2p2s1-face-classification   # the kaggle competition data will be downloaded from
delete_zipfiles_after_unzipping = True                 # whether to delete the zipfiles downloaded from kaggle after unzipping

[wandb]      
wandb_dir = /home/ubuntu/                 # fully qualified directory for wandb internal files
entity = ryanquinnnelson                  # wandb account name
project = CMU-11685-HW2P2A-octopus        # project to save model data under
notes = CNN Face Classification           # any notes to save with this run
tags = CNN,octopus,Resnet34               # any tags to save with this run

[stats]
comparison_metric=val_acc                 # the evaluation metric used to compare this model against previous runs 
comparison_best_is_max=True               # whether a maximum value for the evaluation metric indicates the best performance
comparison_patience=40                    # number of epochs current model can underperform previous runs before early stopping
val_metric_name=val_acc                   # the name of the second metric returned from Evaluation.evalute_model() for clarity in stats

[data]
data_type = image                                                                               # whether data is image type or not (if data_type=image, octopus assumes training and validation data are organized for using ImageFolder)
data_dir = /home/ubuntu/content/competitions/idl-fall21-hw2p2s1-face-classification             # fully qualified root directory where all input data is located
train_dir=/home/ubuntu/content/competitions/idl-fall21-hw2p2s1-face-classification/train_data   # fully qualified directory for training data
val_dir=/home/ubuntu/content/competitions/idl-fall21-hw2p2s1-face-classification/val_data       # fully qualified directory for validation data
test_dir=/home/ubuntu/content/competitions/idl-fall21-hw2p2s1-face-classification/test_data     # fully qualified directory for test data

[output]
output_dir = /home/ubuntu/output         # fully qualified directory where test output should be written

[dataloader]
num_workers=8                                                            # number of workers for use in DataLoader when a GPU is available
pin_memory=True                                                          # whether to use pin memory in DataLoader when a GPU is available
batch_size=256                                                           # batch size regardless of a GPU or CPU
transforms_list=RandomHorizontalFlip,RandomRotation,ToTensor,Normalize   # for image data, the list of transforms to perform and in what order

[model]
model_type=Resnet34_v3     # which predefined model to use for training (check modelhandlers for possible options)

# for Resnet models only
in_features = 3            # for Resnet models, the number of input channels
num_classes = 4000         # for Resnet models performing classification, the number of classes

# for CNN2d models only
input_size=64              # pixel dimension of square images used as input to the model
output_size=4000           # for classification, the number of classes
batch_norm=True            # whether CNN layer should be followed by a batch normalization layer
activation_func=ReLU       # the activation function to use in each CNN block
pool_class=MaxPool2d       # the type of pooling classes used

# for CNN2d models only
# each layer.parameter for the CNN classes or pooling classes 
# 1.padding=1 means the first CNN layer has padding=1
# multiple CNN layers can be listed one after the other (i.e. 1.in_channels=3,..., 2.in_channels=3...)
conv_kwargs=1.in_channels=3,1.out_channels=4,1.kernel_size=3,1.stride=1,1.padding=1,1.dilation=1  
pool_kwargs=1.kernel_size=2,1.stride=2,1.padding=0,1.dilation=1   


[checkpoint]
checkpoint_dir = /data/checkpoints   # fully qualified directory where checkpoints will be saved
delete_existing_checkpoints = False  # whether to delete all checkpoints in the checkpoint directory before starting model training (overridden to False if model is being loaded from a previous checkpoint)
load_from_checkpoint=False           # whether model will be loaded from a previous checkpoint
checkpoint_file = None               # fully qualified checkpoint file

[hyperparameters]
num_epochs = 50                                               # number of epochs to train
criterion_type=CrossEntropyLoss                               # the loss function to use
optimizer_type=SGD                                            # optimizer to use
optimizer_kwargs=lr=0.1,weight_decay=0.00005,momentum=0.5     # any optimizer arguments
scheduler_type=ReduceLROnPlateau                              # the scheduler to use
scheduler_kwargs=factor=0.5,patience=3,mode=max,verbose=True  # any scheduler arguments
scheduler_plateau_metric=val_acc                              # if using a plateau-based scheduler, the evaluation metric tracked by the scheduler 
```

## To customize the code for a new dataset

Given the understanding that certain aspects of a deep learning model must be customized to the dataset, `octopus`
relies on the user having defined a `customized` python module to manage these aspects. To work smoothly with `octopus`
, `customized` must contain the following files and classes. An example `customized` module is provided with this code.

### datasets.py

For image data, if training and validation datasets are formatted to facilitate using `ImageFolder`:

- `TestDataset(Dataset)` class implementing the following methods:
    - `__init__(self, test_dir, transform)`
    - `__len__(self) -> int`
    - `__getitem__(self, index) -> Tensor`

For other data:

- `TrainValDataset(Dataset)` class implementing the following methods:
    - `__init__()`
    - `__len__() -> int`
    - `__getitem__() -> Tensor`


- `TestDataset(Dataset)`class implementing the following methods:
    - `__init__()`
    - `__len__() -> int`
    - `__getitem__() -> Tensor`

### evaluation.py

- `Evaluation` class implementing the following methods:
    - `__init__(self, val_loader, criterion_func, devicehandler)`
    - `evaluate_model(self, epoch, num_epochs, model) -> (val_loss, val_metric)`

### formatters.py

- `OutputFormatter` class implementing the following methods:
    - `__init__(self, data_dir)`
    - `format_output(self, out) -> DataFrame`

