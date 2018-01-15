import pytest
import os
import json
import platform
import tensorflow
import keras

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


@pytest.fixture('function')
def expected_default_options():
    return {
        'batch_size': 1,
        'decay': 0.0005,
        'epochs': 1,
        'loss_function': 'categorical_crossentropy',
        'max_queue_size': 16,
        'metrics': ['accuracy'],
        'model_kwargs': {},
        'momentum': 0.9,
        'num_classes': 2,
        'num_gpus': 1,
        'output_logs_dir': 'redacted',
        'output_model_dir': 'redacted',
        'pooling': 'avg',
        'sgd_lr': 0.01,
        'train_dataset_dir': 'redacted',
        'val_dataset_dir': 'redacted',
        'verbose': False,
        'weights': 'imagenet',
        'workers': 1
    }


def test_mobilenet_v1_on_catdog_datasets_with_missing_required_options(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir):
    with pytest.raises(ValueError, message='missing required option: model_spec'):
        Trainer(
            train_dataset_dir=catdog_train_dataset_dir,
            val_dataset_dir=catdog_val_dataset_dir,
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            num_classes=len(catdog_dictionary),
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 1.0}
        )


def test_mobilenet_v1_on_catdog_datasets_with_extra_unsupported_options(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir):
    with pytest.raises(ValueError, message='unsupported options given: some_other_arg'):
        Trainer(
            some_other_arg='foo',
            model_spec='mobilenet_v1',
            train_dataset_dir=catdog_train_dataset_dir,
            val_dataset_dir=catdog_val_dataset_dir,
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            num_classes=len(catdog_dictionary),
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 1.0}
        )


def test_mobilenet_v1_on_catdog_datasets(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir, expected_default_options):
    trainer = Trainer(
        model_spec='mobilenet_v1',
        train_dataset_dir=catdog_train_dataset_dir,
        val_dataset_dir=catdog_val_dataset_dir,
        output_model_dir=output_model_dir,
        output_logs_dir=output_logs_dir,
        num_classes=len(catdog_dictionary),
        epochs=1,
        batch_size=1,
        model_kwargs={'alpha': 1.0}
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert len(actual) == 3

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')

    with open(os.path.join(output_model_dir, 'training.json')) as file:
        actual = json.loads(file.read())
    expected = {
        'versions': {
            'python': platform.python_version(),
            'tensorflow': tensorflow.__version__,
            'keras': keras.__version__
        },
        'options': expected_default_options
    }
    expected['options']['model_kwargs'] = {'alpha': 1.0}
    expected['options']['model_spec'] = {
        'klass': 'keras.applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    }
    actual['options']['output_logs_dir'] = 'redacted'
    actual['options']['output_model_dir'] = 'redacted'
    actual['options']['train_dataset_dir'] = 'redacted'
    actual['options']['val_dataset_dir'] = 'redacted'
    assert actual == expected


def test_mobilenet_v1_on_catdog_datasets_with_model_spec_override(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir, expected_default_options):
    model_spec = ModelSpec.get(
        'mobilenet_v1',
        klass='keras.applications.mobilenet.MobileNet',
        target_size=[512, 512, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )
    trainer = Trainer(
        model_spec=model_spec,
        train_dataset_dir=catdog_train_dataset_dir,
        val_dataset_dir=catdog_val_dataset_dir,
        output_model_dir=output_model_dir,
        output_logs_dir=output_logs_dir,
        num_classes=len(catdog_dictionary),
        epochs=1,
        batch_size=1,
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert len(actual) == 3

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')

    with open(os.path.join(output_model_dir, 'training.json')) as file:
        actual = json.loads(file.read())
    expected = {
        'versions': {
            'python': platform.python_version(),
            'tensorflow': tensorflow.__version__,
            'keras': keras.__version__
        },
        'options': expected_default_options
    }
    expected['options']['model_spec'] = {
        'klass': 'keras.applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': [1, 2, 3],
        'preprocess_func': 'mean_subtraction',
        'target_size': [512, 512, 3]
    }
    actual['options']['output_logs_dir'] = 'redacted'
    actual['options']['output_model_dir'] = 'redacted'
    actual['options']['train_dataset_dir'] = 'redacted'
    actual['options']['val_dataset_dir'] = 'redacted'
    assert actual == expected


def test_mobilenet_v1_on_catdog_datasets_with_num_gpus_override(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir, expected_default_options):
    model_spec = ModelSpec.get(
        'mobilenet_v1',
        klass='keras.applications.mobilenet.MobileNet',
        target_size=[512, 512, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )
    trainer = Trainer(
        model_spec=model_spec,
        train_dataset_dir=catdog_train_dataset_dir,
        val_dataset_dir=catdog_val_dataset_dir,
        output_model_dir=output_model_dir,
        output_logs_dir=output_logs_dir,
        num_classes=len(catdog_dictionary),
        epochs=1,
        batch_size=1,
        num_gpus=4
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert len(actual) == 3

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')

    with open(os.path.join(output_model_dir, 'training.json')) as file:
        actual = json.loads(file.read())
    expected = {
        'versions': {
            'python': platform.python_version(),
            'tensorflow': tensorflow.__version__,
            'keras': keras.__version__
        },
        'options': expected_default_options
    }
    expected['options']['num_gpus'] = 4
    expected['options']['model_spec'] = {
        'klass': 'keras.applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': [1, 2, 3],
        'preprocess_func': 'mean_subtraction',
        'target_size': [512, 512, 3]
    }
    actual['options']['output_logs_dir'] = 'redacted'
    actual['options']['output_model_dir'] = 'redacted'
    actual['options']['train_dataset_dir'] = 'redacted'
    actual['options']['val_dataset_dir'] = 'redacted'
    assert actual == expected


def test_resnet50_on_catdog_datasets(catdog_dictionary, catdog_train_dataset_dir, catdog_val_dataset_dir, output_model_dir, output_logs_dir, expected_default_options):
    trainer = Trainer(
        model_spec=ModelSpec.get('resnet50', preprocess_args=[1, 2, 3]),
        train_dataset_dir=catdog_train_dataset_dir,
        val_dataset_dir=catdog_val_dataset_dir,
        output_model_dir=output_model_dir,
        output_logs_dir=output_logs_dir,
        num_classes=len(catdog_dictionary),
        epochs=1,
        batch_size=1,
    )
    trainer.run()

    actual = list_files(output_model_dir, relative=True)
    assert len(actual) == 3

    actual = list_files(output_logs_dir, relative=True)
    assert len(actual) == 2
    for path in actual:
        assert path.startswith('events.out.tfevents.')

    with open(os.path.join(output_model_dir, 'training.json')) as file:
        actual = json.loads(file.read())
    expected = {
        'versions': {
            'python': platform.python_version(),
            'tensorflow': tensorflow.__version__,
            'keras': keras.__version__
        },
        'options': expected_default_options
    }
    expected['options']['model_spec'] = {
        'klass': 'keras.applications.resnet50.ResNet50',
        'name': 'resnet50',
        'preprocess_args': [1, 2, 3],
        'preprocess_func': 'mean_subtraction',
        'target_size': [224, 224, 3]
    }
    actual['options']['output_logs_dir'] = 'redacted'
    actual['options']['output_model_dir'] = 'redacted'
    actual['options']['train_dataset_dir'] = 'redacted'
    actual['options']['val_dataset_dir'] = 'redacted'
    assert actual == expected
