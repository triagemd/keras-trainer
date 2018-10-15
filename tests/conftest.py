import os
import pytest


@pytest.fixture('session')
def train_catdog_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'train'))


@pytest.fixture('session')
def val_catdog_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'val'))
