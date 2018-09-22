import pytest
import os
import json
import platform
import tensorflow
import keras
import numpy as np

from backports.tempfile import TemporaryDirectory
from stored import list_files
from keras_model_specs import ModelSpec
from keras_trainer import Trainer
from keras_applications import mobilenet
from keras_trainer.dataloaders import BalancedImageDataGenerator, ImageDataGeneratorSameMultiGT
from keras_trainer.losses import entropy_penalty_loss


def check_train_on_catdog_datasets(trainer_args={}, expected_model_spec={}, expected_model_files=5, check_opts=True):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=os.path.abspath('tests/files/catdog/train'),
            val_dataset_dir=os.path.abspath('tests/files/catdog/val'),
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 1.0},
            **trainer_args
        )
        trainer.run()

        actual = list_files(output_model_dir, relative=True)
        assert len(actual) == expected_model_files

        actual = list_files(output_logs_dir, relative=True)
        assert len(actual) == 2
        for path in actual:
            assert path.startswith('events.out.tfevents.')

        with open(os.path.join(output_model_dir, 'training_options.json')) as file:
            actual = json.loads(file.read())
        actual['options']['output_logs_dir'] = 'redacted'
        actual['options']['output_model_dir'] = 'redacted'
        actual['options']['train_dataset_dir'] = 'redacted'
        actual['options']['val_dataset_dir'] = 'redacted'

        expected = {
            'versions': {
                'python': platform.python_version(),
                'tensorflow': tensorflow.__version__,
                'keras': keras.__version__
            },
            'options': {
                'batch_size': 1,
                'decay': 0.0005,
                'epochs': 1,
                'loss_function': 'categorical_crossentropy',
                'max_queue_size': 16,
                'metrics': ['accuracy'],
                'model_kwargs': {'alpha': 1.0},
                'momentum': 0.9,
                'num_gpus': 0,
                'output_logs_dir': 'redacted',
                'output_model_dir': 'redacted',
                'pooling': 'avg',
                'include_top': False,
                'sgd_lr': 0.01,
                'dropout_rate': 0.0,
                'activation': 'softmax',
                'train_dataset_dir': 'redacted',
                'val_dataset_dir': 'redacted',
                'verbose': False,
                'weights': 'imagenet',
                'workers': 1,
            }
        }

        expected['options'].update(trainer_args)
        expected['options']['model_spec'] = expected_model_spec

        if check_opts:
            assert actual == expected


def check_freeze_layers_train_on_catdog_datasets(trainer_args={}, expected_model_spec={}, expected_model_files=5, check_opts=True):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=os.path.abspath('tests/files/catdog/train'),
            val_dataset_dir=os.path.abspath('tests/files/catdog/val'),
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 1.0},
            freeze_layers_list=list(range(1, 10)),
            **trainer_args
        )
        trainer.run()

        actual = trainer.model.layers[5].trainable
        expected = False

        assert actual == expected


def check_freeze_layers_train_on_catdog_datasets_with_float(trainer_args={}, expected_model_spec={}, expected_model_files=5, check_opts=True):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=os.path.abspath('tests/files/catdog/train'),
            val_dataset_dir=os.path.abspath('tests/files/catdog/val'),
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 1.0},
            freeze_layers_list=np.arange(1, 10, 0.5),
            **trainer_args
        )
        trainer.run()


def test_custom_model_on_catdog_datasets():
    model = mobilenet.MobileNet(alpha=0.25, include_top=False, pooling='avg', input_shape=[224, 224, 3])
    top_layers = []
    # Set Dense Layer
    top_layers.append(keras.layers.Dense(2, name='dense'))
    # Set Activation Layer
    top_layers.append(keras.layers.Activation('softmax', name='act_softmax'))

    # Layer Assembling
    for i, layer in enumerate(top_layers):
        if i == 0:
            top_layers[i] = layer(model.output)
        else:
            top_layers[i] = layer(top_layers[i - 1])

    # Final Model (last item of self.top_layer contains all of them assembled)
    model = keras.models.Model(model.input, top_layers[-1])
    check_train_on_catdog_datasets({'custom_model': model,
                                    'model_spec': ModelSpec.get('mobilenet_custom', preprocess_args=[1, 2, 3],
                                                                preprocess_func='mean_subtraction',
                                                                target_size=[224, 224, 3])
                                    },
                                   {
                                       'klass': None,
                                       'name': 'mobilenet_custom',
                                       'preprocess_args': [1, 2, 3],
                                       'preprocess_func': 'mean_subtraction',
                                       'target_size': [224, 224, 3]
    },
        check_opts=False
    )


def test_freeze_layers_on_catdog_datasets():
    check_freeze_layers_train_on_catdog_datasets({
        'model_spec': 'mobilenet_v1'
    }, {
        'klass': 'keras_applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    })
    with pytest.raises(ValueError, message="<class 'numpy.float64'> layer type not supported to freeze layers"):
        check_freeze_layers_train_on_catdog_datasets_with_float({
            'model_spec': 'mobilenet_v1'
        }, {
            'klass': 'keras_applications.mobilenet.MobileNet',
            'name': 'mobilenet_v1',
            'preprocess_args': None,
            'preprocess_func': 'between_plus_minus_1',
            'target_size': [224, 224, 3]
        })


def test_custom_model_on_catdog_datasets_with_multi_loss():
    model = mobilenet.MobileNet(alpha=0.25, include_top=False, pooling='avg', input_shape=[224, 224, 3])
    top_layers = []
    # Set Dense Layer
    top_layers.append(keras.layers.Dense(2, name='dense'))
    # Set Activation Layer
    top_layers.append(keras.layers.Activation('softmax', name='act_softmax'))

    # Layer Assembling
    for i, layer in enumerate(top_layers):
        if i == 0:
            top_layers[i] = layer(model.output)
        else:
            top_layers[i] = layer(top_layers[i - 1])

    # Final Model (last item of self.top_layer contains all of them assembled)
    model = keras.models.Model(model.input, [top_layers[-1], top_layers[-1]])

    check_train_on_catdog_datasets({'custom_model': model,
                                    'train_data_generator': ImageDataGeneratorSameMultiGT(n_outputs=2),
                                    'val_data_generator': ImageDataGeneratorSameMultiGT(n_outputs=2),
                                    'loss_function': ['categorical_crossentropy', entropy_penalty_loss],
                                    'loss_weights': [1.0, 0.25],
                                    'model_spec': ModelSpec.get('mobilenet_custom_2_outputs', preprocess_args=[1, 2, 3],
                                                                preprocess_func='mean_subtraction',
                                                                target_size=[224, 224, 3])
                                    },
                                   {
                                       'klass': None,
                                       'name': 'mobilenet_custom_2_outputs',
                                       'preprocess_args': [1, 2, 3],
                                       'preprocess_func': 'mean_subtraction',
                                       'target_size': [224, 224, 3]
    },
        check_opts=False
    )


def test_mobilenet_v1_on_catdog_datasets_with_balanced_generator():
    check_train_on_catdog_datasets({
        'train_data_generator': BalancedImageDataGenerator(),
        'model_spec': 'mobilenet_v1'
    }, {
        'klass': 'keras_applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    },
        check_opts=False
    )


def test_mobilenet_v1_on_catdog_datasets_with_missing_required_options():
    with pytest.raises(ValueError, message='missing required option: model_spec'):
        check_train_on_catdog_datasets()


def test_mobilenet_v1_on_catdog_datasets_with_extra_unsupported_options():
    with pytest.raises(ValueError, message='unsupported options given: some_other_arg'):
        check_train_on_catdog_datasets({
            'model_spec': 'mobilenet_v1',
            'some_other_arg': 'foo'
        })


def test_mobilenet_v1_on_catdog_datasets_with_dropout():
    check_train_on_catdog_datasets({
        'dropout_rate': 0.5,
        'model_spec': 'mobilenet_v1'
    }, {
        'klass': 'keras_applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    })


def test_mobilenet_v1_on_catdog_datasets():
    check_train_on_catdog_datasets({
        'model_spec': 'mobilenet_v1'
    }, {
        'klass': 'keras_applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    })


def test_mobilenet_v1_on_catdog_datasets_with_model_spec_override():
    check_train_on_catdog_datasets({
        'model_spec': ModelSpec.get(
            'mobilenet_v1',
            klass='keras_applications.mobilenet.MobileNet',
            target_size=[512, 512, 3],
            preprocess_func='mean_subtraction',
            preprocess_args=[1, 2, 3]
        )
    }, {
        'klass': 'keras_applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': [1, 2, 3],
        'preprocess_func': 'mean_subtraction',
        'target_size': [512, 512, 3]
    })


def test_resnet50_on_catdog_datasets():
    check_train_on_catdog_datasets({
        'model_spec': ModelSpec.get('resnet50', preprocess_args=[1, 2, 3])
    }, {
        'klass': 'keras_applications.resnet50.ResNet50',
        'name': 'resnet50',
        'preprocess_args': [1, 2, 3],
        'preprocess_func': 'mean_subtraction',
        'target_size': [224, 224, 3]
    })
