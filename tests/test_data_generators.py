import numpy as np
import pandas as pd

from PIL import Image
from keras_trainer.data_generators import EnhancedDirectoryIterator, EnhancedImageDataGenerator, \
    EnhancedDataFrameIterator

# Core functions are already tested in https://github.com/keras-team/keras-preprocessing/blob/master/tests/image_test.py
# Here we provide tests for the functionalities added


def test_enhanced_directory_iterator_mode_equiprobable(train_catdog_dataset_path, train_catdog_dataset_json_path):
    dataframe = pd.read_json(train_catdog_dataset_json_path)
    iterators = [EnhancedDirectoryIterator(train_catdog_dataset_path,
                                           EnhancedImageDataGenerator(),
                                           iterator_mode='equiprobable',
                                           batch_size=2),
                 EnhancedDataFrameIterator(dataframe,
                                           EnhancedImageDataGenerator(),
                                           directory=train_catdog_dataset_path,
                                           x_col="filename",
                                           y_col="class_probabilities",
                                           iterator_mode='equiprobable',
                                           batch_size=2)
                 ]

    for iterator in iterators:
        x, y = iterator.next()

        assert x.shape == (2, 256, 256, 3)
        assert y.shape == (2, 2)


def test_enhanced_directory_iterator_same_n_outputs(train_catdog_dataset_path, train_catdog_dataset_json_path):
    dataframe = pd.read_json(train_catdog_dataset_json_path)
    iterators = [EnhancedDirectoryIterator(train_catdog_dataset_path,
                                           EnhancedImageDataGenerator(),
                                           n_outputs=2,
                                           batch_size=2),
                 EnhancedDataFrameIterator(dataframe,
                                           EnhancedImageDataGenerator(),
                                           directory=train_catdog_dataset_path,
                                           x_col="filename",
                                           y_col="class_probabilities",
                                           n_outputs=2,
                                           batch_size=2)
                 ]

    for iterator in iterators:
        x, y = iterator.next()

        assert x.shape == (2, 256, 256, 3)
        assert len(y) == 2
        np.testing.assert_array_equal(y[0], y[1])


def test_enhanced_image_data_generator_random_crop(train_catdog_dataset_path, train_catdog_dataset_json_path):
    dataframe = pd.read_json(train_catdog_dataset_json_path)
    generator = EnhancedImageDataGenerator(random_crop_size=0.8)
    datagens = [generator.flow_from_directory(train_catdog_dataset_path,
                                              batch_size=1,
                                              target_size=(256, 256)),
                generator.flow_from_dataframe(dataframe,
                                              directory=train_catdog_dataset_path,
                                              x_col="filename",
                                              y_col="class_probabilities",
                                              batch_size=1,
                                              target_size=(256, 256)),
                ]

    for datagen in datagens:
        x, y = datagen.next()

        assert x.shape == (1, 256, 256, 3)
        assert len(y) == 1

        img = Image.open('tests/files/catdog/val/cat/cat-4.jpg')
        x, y, dx, dy = datagen.random_crop_parameters(img, crop_size=(224, 224), crop_mode='size')

        assert dx - x == 224
        assert dy - y == 224

        x, y, dx, dy = datagen.random_crop_parameters(img, crop_size=0.5, crop_mode='percentage')

        assert dx - x == 665
        assert dy - y == 665

        x, y, dx, dy = datagen.random_crop_parameters(img, crop_size=(55, 200), crop_mode='size')

        assert dx - x == 55
        assert dy - y == 200
