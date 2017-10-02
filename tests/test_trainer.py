import pytest
import os

from backports.tempfile import TemporaryDirectory
from stored import list_files
from keras_model_specs import ModelSpec
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


@pytest.fixture('session')
def catdog_num_classes():
    return 2


def test_resnet50_on_catdog_datasets(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir):
    trainer = Trainer(
        ModelSpec.get('resnet50', preprocess_args=[1, 2, 3]),
        catdog_train_dataset_dir,
        catdog_val_dataset_dir,
        output_model_dir,
        output_logs_dir,
        num_classes=len(catdog_dictionary),
        epochs=1,
        batch_size=1,
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert sorted(actual) == sorted(['best.hdf5', 'final.hdf5'])

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')


def test_mobilenet_v1_on_catdog_datasets(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir):
    trainer = Trainer(
        'mobilenet_v1',
        catdog_train_dataset_dir,
        catdog_val_dataset_dir,
        output_model_dir,
        output_logs_dir,
        num_classes=len(catdog_dictionary),
        epochs=1,
        batch_size=1,
        model_kwargs={'alpha': 1.0}
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert sorted(actual) == sorted(['best.hdf5', 'final.hdf5'])

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')


def test_mobilenet_v1_on_catdog_datasets_with_model_spec_override(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir):
    model_spec = ModelSpec.get(
        'mobilenet_v1',
        klass='keras.applications.mobilenet.MobileNet',
        target_size=[512, 512, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )
    trainer = Trainer(
        model_spec,
        catdog_train_dataset_dir,
        catdog_val_dataset_dir,
        output_model_dir,
        output_logs_dir,
        num_classes=len(catdog_dictionary),
        epochs=1,
        batch_size=1,
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert sorted(actual) == sorted(['best.hdf5', 'final.hdf5'])

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')
