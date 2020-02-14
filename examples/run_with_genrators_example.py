import os
import json
import numpy as np
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras_trainer import Trainer
from keras_trainer.data_generators import EnhancedImageDataGenerator

# Define the path of the train and validation lab manifests and read them
train_dataset_json_path = "/path/to/lab/training/manifest"
val_dataset_json_path = "/path/to/lab/validation/manifest"

train_dataframe = pd.read_json(train_dataset_json_path)
val_dataframe = pd.read_json(val_dataset_json_path)

# Define the path where the images exist in the dgx e.g. "/data/lab/images/files"
root_directory = "/path/of/root/directory"

# Set the class number e.g. 8
class_number = 8

# Choose the model architecture and target size
architecture = 'inception_v3'
target_size = 299

# Define some arguments to the Trainer object
trainer_args = {
    'train_generator': EnhancedImageDataGenerator().flow_from_dataframe(
        train_dataframe,
        x_col="storage_key",
        y_col="class_probabilities",
        target_size=(target_size, target_size),
        validate_filenames=False
    ),
    'model_spec': architecture,
    'num_classes': class_number,
    'val_generator': EnhancedImageDataGenerator().flow_from_dataframe(
            val_dataframe,
            x_col="storage_key",
            y_col="class_probabilities",
            target_size=(target_size, target_size),
            validate_filenames=False
    )
}

# Create the Trainer object and run it
trainer = Trainer(
            train_dataset_dir=root_directory,
            val_dataset_dir=root_directory,
            output_model_dir="./{}_model_dir".format(architecture),
            output_logs_dir="./{}_log_dir".format(architecture),
            epochs=50,
            num_gpus=3,
            batch_size=32,
            model_kwargs={'alpha': 0.25},
            **trainer_args)

trainer.run()
