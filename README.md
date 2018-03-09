[![Build Status](https://travis-ci.org/triagemd/keras-trainer.svg?branch=master)](https://travis-ci.org/triagemd/keras-trainer) [![PyPI version](https://badge.fury.io/py/keras-trainer.svg)](https://badge.fury.io/py/keras-trainer)

## Keras Trainer

An abstraction to train Keras CNN models for image classification. To use it is required to have also installed the `keras-model-specs` package.

The list of models supported is the following:

`vgg16`, `vgg19`, `resnet50`, `resnet152`, `mobilenet_v1`, `xception`,
`inception_resnet_v2`, `inception_v3`, `inception_v4`, `nasnet_large`, `nasnet_mobile`, `densenet_169`,
`densenet_121`, `densenet_201`

And the defaults are specified [here](https://github.com/triagemd/keras-model-specs/blob/master/keras_model_specs/model_specs.json).

### Keras Trainer definition

These are the default options:

```
OPTIONS = {
        'model_spec': {'type': str},
        'train_dataset_dir': {'type': str},
        'val_dataset_dir': {'type': str},
        'output_model_dir': {'type': str},
        'output_logs_dir': {'type': str},
        'include_top': {'type': bool, 'default': False},
        'input_shape': {'type': None, 'default': None},
        'checkpoint_path': {'type': str, 'default': None},
        'train_data_generator': {'type': None, 'default': None},
        'val_data_generator': {'type': None, 'default': None},
        'train_generator': {'type': None, 'default': None},
        'val_generator': {'type': None, 'default': None},
        'top_layers': {'type': None, 'default': None},
        'optimizer': {'type': None, 'default': None},
        'callback_list': {'type': list, 'default': []},
        'class_weights': {'type': None, 'default': None},
        'loss_function': {'type': str, 'default': 'categorical_crossentropy'},
        'metrics': {'type': list, 'default': ['accuracy']},
        'batch_size': {'type': int, 'default': 1},
        'epochs': {'type': int, 'default': 1},
        'decay': {'type': float, 'default': 0.0005},
        'momentum': {'type': float, 'default': 0.9},
        'sgd_lr': {'type': float, 'default': 0.01},
        'pooling': {'type': str, 'default': 'avg'},
        'activation': {'type': str, 'default': 'softmax'},
        'dropout_rate': {'type': float, 'default': 0.0},
        'freeze_layers_list': {'type': None, 'default': None},
        'weights': {'type': str, 'default': 'imagenet'},
        'num_gpus': {'type': int, 'default': 0},
        'workers': {'type': int, 'default': 1},
        'max_queue_size': {'type': int, 'default': 16},
        'num_classes': {'type': int, 'default': None},
        'verbose': {'type': str, 'default': False},
        'model_kwargs': {'type': dict, 'default': {}}
    }
```

You will find a guide of use [here](https://github.com/triagemd/keras-trainer/blob/master/example.ipynb)
