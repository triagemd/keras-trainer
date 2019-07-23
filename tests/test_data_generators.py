import numpy as np

from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from keras_trainer.data_generators import BalancedDirectoryIterator, DirectoryIteratorSameMultiGT, \
    RandomCropImageDataGenerator, RandomCropDirectoryIterator

# Core functions are already tested in https://github.com/keras-team/keras-preprocessing/blob/master/tests/image_test.py
# Here we provide tests for the functionalities added


def test_balanced_directory_iterator(train_catdog_dataset_path):
    iterator = BalancedDirectoryIterator(train_catdog_dataset_path,
                                         ImageDataGenerator(),
                                         batch_size=2)
    x, y = iterator.next()

    assert x.shape == (2, 256, 256, 3)
    assert y.shape == (2, 2)


def test_directory_iterator_same_multi_gt(train_catdog_dataset_path):
    iterator = DirectoryIteratorSameMultiGT(train_catdog_dataset_path,
                                            ImageDataGenerator(),
                                            n_outputs=2,
                                            batch_size=2)
    x, y = iterator.next()

    assert x.shape == (2, 256, 256, 3)
    assert len(y) == 2
    np.testing.assert_array_equal(y[0], y[1])


def test_random_crop_image_data_generator(train_catdog_dataset_path):
    generator = RandomCropImageDataGenerator(crop_size=0.8)
    datagen = generator.flow_from_directory(train_catdog_dataset_path,
                                            batch_size=1,
                                            target_size=(256, 256))

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
