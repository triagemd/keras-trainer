import os
import json
import keras
import pytest
import platform
import tensorflow
import numpy as np
import pandas as pd

from stored import list_files
from keras_trainer import Trainer
from keras.regularizers import l2
from keras_model_specs import ModelSpec
from backports.tempfile import TemporaryDirectory
from keras_trainer.losses import entropy_penalty_loss
from keras_trainer.data_generators import EnhancedImageDataGenerator


def check_train_on_catdog_datasets(train_path, val_path, trainer_args={}, expected_model_spec={},
                                   expected_model_files=5, check_opts=True):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=train_path,
            val_dataset_dir=val_path,
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 0.25},
            **trainer_args
        )
        trainer.run()

        actual = list_files(output_model_dir, relative=True)
        assert len(actual) == expected_model_files

        actual = list_files(output_logs_dir, relative=True)

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
                'model_kwargs': {'alpha': 0.25},
                'momentum': 0.9,
                'num_gpus': 0,
                'custom_crop': False,
                'output_logs_dir': 'redacted',
                'output_model_dir': 'redacted',
                'track_sensitivity': False,
                'pooling': 'avg',
                'save_training_options': True,
                'include_top': False,
                'sgd_lr': 0.01,
                'dropout_rate': 0.0,
                'regularize_bias': False,
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


def check_freeze_layers_train_on_catdog_datasets_int(train_path, val_path, trainer_args={}):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=train_path,
            val_dataset_dir=val_path,
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 0.25},
            freeze_layers_list=list(range(1, 10)),
            **trainer_args
        )
        trainer.run()

        for i in range(1, 10):
            actual = trainer.model.layers[i].trainable
            expected = False

            assert actual == expected


def check_freeze_layers_train_on_catdog_datasets_np_int(train_path, val_path, trainer_args={}):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=train_path,
            val_dataset_dir=val_path,
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 0.25},
            freeze_layers_list=list(np.arange(1, 10)),
            **trainer_args
        )

        # Tested in other function
        # trainer.run()

        for i in range(1, 10):
            actual = trainer.model.layers[i].trainable
            expected = False

            assert actual == expected


def check_freeze_layers_train_on_catdog_datasets_str(train_path, val_path, freeze_layers_list_str, trainer_args={}):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=train_path,
            val_dataset_dir=val_path,
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 0.25},
            freeze_layers_list=freeze_layers_list_str,
            **trainer_args
        )

        # Tested in other function
        # trainer.run()

        for layer in freeze_layers_list_str:
            actual = trainer.model.get_layer(layer).trainable
            expected = False

            assert actual == expected


def check_freeze_layers_train_on_catdog_datasets_with_float(train_path, val_path, trainer_args={}):
    with TemporaryDirectory() as output_model_dir, TemporaryDirectory() as output_logs_dir:
        trainer = Trainer(
            train_dataset_dir=train_path,
            val_dataset_dir=val_path,
            output_model_dir=output_model_dir,
            output_logs_dir=output_logs_dir,
            epochs=1,
            batch_size=1,
            model_kwargs={'alpha': 0.25},
            freeze_layers_list=np.arange(1, 10, 0.5),
            **trainer_args
        )
        trainer.run()


def test_custom_model_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path, simple_model):
    trainer_args = {'custom_model': simple_model,
                    'model_spec': ModelSpec.get('custom', preprocess_args=[1, 2, 3],
                                                preprocess_func='mean_subtraction',
                                                target_size=[224, 224, 3])
                    }
    expected_model_spec = {'model': None,
                           'name': 'custom',
                           'preprocess_args': [1, 2, 3],
                           'preprocess_func': 'mean_subtraction',
                           'target_size': [224, 224, 3]
                           }

    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path, trainer_args,
                                   expected_model_spec, check_opts=False)


def test_custom_model_track_sensitivity_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                                           simple_model):
    trainer_args = {'custom_model': simple_model,
                    'model_spec': ModelSpec.get('custom', preprocess_args=[1, 2, 3],
                                                preprocess_func='mean_subtraction',
                                                target_size=[224, 224, 3]),
                    'track_sensitivity': True
                    }
    expected_model_spec = {'model': None,
                           'name': 'custom',
                           'preprocess_args': [1, 2, 3],
                           'preprocess_func': 'mean_subtraction',
                           'target_size': [224, 224, 3],
                           }

    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path, trainer_args,
                                   expected_model_spec, check_opts=False, expected_model_files=6)


def test_freeze_layers_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path):
    freeze_layer_lst = ['conv1', 'conv1_bn']
    trainer_args = {'model_spec': 'mobilenet_v1'}

    check_freeze_layers_train_on_catdog_datasets_int(train_catdog_dataset_path, val_catdog_dataset_path,
                                                     trainer_args)

    check_freeze_layers_train_on_catdog_datasets_np_int(train_catdog_dataset_path, val_catdog_dataset_path,
                                                        trainer_args)

    check_freeze_layers_train_on_catdog_datasets_str(train_catdog_dataset_path, val_catdog_dataset_path,
                                                     freeze_layer_lst, trainer_args)

    with pytest.raises(ValueError, match="<class 'numpy.float64'> layer type not supported to freeze layers, "
                       "we expect an int giving the layer index or a str containing "
                       "the name of the layer."):
        check_freeze_layers_train_on_catdog_datasets_with_float(train_catdog_dataset_path, val_catdog_dataset_path,
                                                                trainer_args)


def test_custom_model_on_catdog_datasets_with_multi_loss(train_catdog_dataset_path, val_catdog_dataset_path,
                                                         simple_model):
    model = keras.models.Model(simple_model.input, [simple_model.output, simple_model.output])

    trainer_args = {'custom_model': model,
                    'loss_function': ['categorical_crossentropy', entropy_penalty_loss],
                    'loss_weights': [1.0, 0.25],
                    'model_spec': ModelSpec.get('model_custom_2_outputs', preprocess_args=[1, 2, 3],
                                                preprocess_func='mean_subtraction',
                                                target_size=[224, 224, 3])
                    }
    expected_model_spec = {'model': None,
                           'name': 'model_custom_2_outputs',
                           'preprocess_args': [1, 2, 3],
                           'preprocess_func': 'mean_subtraction',
                           'target_size': [224, 224, 3]
                           }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec, check_opts=False)


def test_custom_model_on_catdog_datasets_with_probabilistic_labels(train_catdog_dataset_path, val_catdog_dataset_path,
                                                                   train_catdog_dataset_json_path,
                                                                   val_catdog_dataset_json_path,
                                                                   simple_model):
    train_dataframe = pd.read_json(train_catdog_dataset_json_path)
    val_dataframe = pd.read_json(val_catdog_dataset_json_path)
    trainer_args = {
        'custom_model': simple_model,
        'train_generator': EnhancedImageDataGenerator().flow_from_dataframe(
            train_dataframe,
            directory=train_catdog_dataset_path,
            x_col="filename",
            y_col="class_probabilities",
            crop_col="crop",
            target_size=(224, 224)
        ),
        'model_spec': 'mobilenet_v1',
        'num_classes': 2,
        'val_generator': EnhancedImageDataGenerator().flow_from_dataframe(
            val_dataframe,
            directory=val_catdog_dataset_path,
            x_col="filename",
            y_col="class_probabilities",
            crop_col="crop",
            target_size=(224, 224)
        ),
    }

    expected_model_spec = {'model': None,
                           'name': 'model_custom_2_outputs',
                           'preprocess_args': [1, 2, 3],
                           'preprocess_func': 'mean_subtraction',
                           'target_size': [224, 224, 3]
                           }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec, check_opts=False)


def test_custom_model_on_catdog_datasets_with_mix_directory_dataframe(train_catdog_dataset_path,
                                                                      val_catdog_dataset_path,
                                                                      train_catdog_dataset_json_path,
                                                                      simple_model):
    train_dataframe = pd.read_json(train_catdog_dataset_json_path)
    trainer_args = {
        'custom_model': simple_model,
        'train_generator': EnhancedImageDataGenerator().flow_from_dataframe(
            train_dataframe,
            directory=train_catdog_dataset_path,
            x_col="filename",
            y_col="class_probabilities",
            target_size=(224, 224)
        ),
        'model_spec': 'mobilenet_v1',
        'num_classes': 2,
    }

    expected_model_spec = {'model': None,
                           'name': 'model_custom_2_outputs',
                           'preprocess_args': [1, 2, 3],
                           'preprocess_func': 'mean_subtraction',
                           'target_size': [224, 224, 3]
                           }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec, check_opts=False)


def test_custom_model_on_catdog_datasets_with_probabilistic_labels_from_constructor(train_catdog_dataset_path,
                                                                                    val_catdog_dataset_path,
                                                                                    train_catdog_dataset_json_path,
                                                                                    val_catdog_dataset_json_path,
                                                                                    simple_model):
    trainer_args = {
        'custom_model': simple_model,
        'train_dataset_dataframe_path': train_catdog_dataset_json_path,
        'val_dataset_dataframe_path': val_catdog_dataset_json_path,
        'model_spec': 'mobilenet_v1',
        'num_classes': 2,
    }

    expected_model_spec = {'model': None,
                           'name': 'model_custom_2_outputs',
                           'preprocess_args': [1, 2, 3],
                           'preprocess_func': 'mean_subtraction',
                           'target_size': [224, 224, 3]
                           }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec, check_opts=False)


def test_mobilenet_v1_on_catdog_datasets_with_balanced_generator(train_catdog_dataset_path, val_catdog_dataset_path):
    trainer_args = {
        'train_generator': EnhancedImageDataGenerator().flow_from_directory(
            train_catdog_dataset_path,
            iterator_mode='equiprobable',
            target_size=(224, 224)
        ),
        'model_spec': 'mobilenet_v1',
        'num_classes': 2
    }
    expected_model_spec = {'model': 'keras_applications.mobilenet.MobileNet',
                           'name': 'mobilenet_v1',
                           'preprocess_args': None,
                           'preprocess_func': 'between_plus_minus_1',
                           'target_size': [224, 224, 3]
                           }

    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path, trainer_args,
                                   expected_model_spec, check_opts=False)


def test_mobilenet_v1_on_catdog_datasets_with_missing_required_options(train_catdog_dataset_path, val_catdog_dataset_path):
    with pytest.raises(ValueError, match='missing required option: model_spec'):
        check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path)


def test_mobilenet_v1_on_catdog_datasets_with_extra_unsupported_options(train_catdog_dataset_path, val_catdog_dataset_path):
    with pytest.raises(ValueError, match='unsupported options given: some_other_arg'):
        trainer_args = {'model_spec': 'mobilenet_v1',
                        'some_other_arg': 'foo'
                        }
        check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path, trainer_args=trainer_args)


def test_mobilenet_v1_on_catdog_datasets_with_dropout(train_catdog_dataset_path, val_catdog_dataset_path):
    trainer_args = {'dropout_rate': 0.5,
                    'model_spec': 'mobilenet_v1'
                    }
    expected_model_spec = {'model': 'keras_applications.mobilenet.MobileNet',
                           'name': 'mobilenet_v1',
                           'preprocess_args': None,
                           'preprocess_func': 'between_plus_minus_1',
                           'target_size': [224, 224, 3]
                           }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec)


def test_mobilenet_v1_on_catdog_datasets_with_l2_regularization(train_catdog_dataset_path, val_catdog_dataset_path):
    trainer_args = {'regularization_function': l2(0.00025),
                    'model_spec': 'mobilenet_v1'
                    }
    expected_model_spec = {'model': 'keras_applications.mobilenet.MobileNet',
                           'name': 'mobilenet_v1',
                           'preprocess_args': None,
                           'preprocess_func': 'between_plus_minus_1',
                           'target_size': [224, 224, 3]
                           }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec, check_opts=False)


def test_mobilenet_v1_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path):
    trainer_args = {'model_spec': 'mobilenet_v1'}
    expected_model_spec = {
        'model': 'keras_applications.mobilenet.MobileNet',
        'name': 'mobilenet_v1',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec)


def test_mobilenet_v1_on_catdog_datasets_with_model_spec_override(train_catdog_dataset_path, val_catdog_dataset_path):
    trainer_args = {'model_spec': ModelSpec.get(
        'mobilenet_v1',
        model='keras_applications.mobilenet.MobileNet',
        target_size=[112, 112, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )}
    expected_model_spec = {'model': 'keras_applications.mobilenet.MobileNet',
                           'name': 'mobilenet_v1',
                           'preprocess_args': [1, 2, 3],
                           'preprocess_func': 'mean_subtraction',
                           'target_size': [112, 112, 3]
                           }
    check_train_on_catdog_datasets(train_catdog_dataset_path, val_catdog_dataset_path,
                                   trainer_args, expected_model_spec)
