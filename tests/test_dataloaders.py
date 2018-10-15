import numpy as np

from keras_preprocessing.image import ImageDataGenerator
from keras_trainer.dataloaders import BalancedDirectoryIterator, DirectoryIteratorSameMultiGT

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
