import os
import numpy as np
from keras_trainer.dataloaders import BalancedDirectoryIterator, DirectoryIteratorSameMultiGT
from keras_preprocessing.image import ImageDataGenerator


# Core functions are already tested in https://github.com/keras-team/keras-preprocessing/blob/master/tests/image_test.py
# Here we provide tests for the functionalities added

def test_balanced_directory_iterator():
    iterator = BalancedDirectoryIterator(os.path.abspath('tests/files/catdog/train'),
                                         ImageDataGenerator(),
                                         batch_size=2)
    x, y = iterator.next()

    assert x.shape == (2, 256, 256, 3)
    assert y.shape == (2, 2)


def test_directory_iterator_same_multi_gt():
    iterator = DirectoryIteratorSameMultiGT(os.path.abspath('tests/files/catdog/train'),
                                            ImageDataGenerator(),
                                            n_outputs=2,
                                            batch_size=2)
    x, y = iterator.next()

    assert x.shape == (2, 256, 256, 3)
    assert len(y) == 2
    np.testing.assert_array_equal(y[0], y[1])
