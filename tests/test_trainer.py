import pytest
import os

from backports.tempfile import TemporaryDirectory
from ml_tools import list_files

from keras_trainer import Trainer


@pytest.fixture('session')
def catdog_dictionary():
    return ['cat', 'dog']


@pytest.fixture('session')
def catdog_train_dataset_dir():
    return os.path.abspath('tests/files/catdog/train')


@pytest.fixture('session')
def catdog_val_dataset_dir():
    return os.path.abspath('tests/files/catdog/val')


@pytest.fixture('function')
def output_model_dir():
    with TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture('function')
def output_logs_dir():
    with TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_mobilenet_v1_on_catdog_datasets(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir):
    trainer = Trainer(
        'mobilenet_v1',
        catdog_dictionary,
        catdog_train_dataset_dir,
        catdog_val_dataset_dir,
        output_model_dir,
        output_logs_dir,
        epochs=1,
        batch_size=1,
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert actual == ['best.hdf5', 'final.hdf5']

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')
