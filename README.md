[![Build Status](https://travis-ci.org/triagemd/keras-trainer.svg?branch=master)](https://travis-ci.org/triagemd/keras-trainer) [![PyPI version](https://badge.fury.io/py/keras-trainer.svg)](https://badge.fury.io/py/keras-trainer)

## Keras Trainer

An abstraction to train Keras CNN models for image classification. 

This package provides an easy framework to apply extra training steps such as **image preprocessing**, **random cropping**, **balanced sampling**, training with **probabilistic labels** and using **multi-loss functions**.



The models supported are the ones specified in the [keras-model-specs](https://github.com/triagemd/keras-model-specs) package 
that correspond to the latest models available in [keras-applications](https://github.com/keras-team/keras-applications)

These are:

VGG:

- `vgg16`, `vgg19`

ResNet 

- `resnet50`, `resnet101` `resnet152`, `resnet50_v2`, `resnet101_v2`, `resnet152_v2`

ResNeXt

- `ResNeXt50`, `ResNeXt101`

MobileNet

- `mobilenet_v1`, `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`

Inception

- `inception_resnet_v2`, `inception_v3`

Xception

- `xception`

NasNet

- `nasnet_large`, `nasnet_mobile`

DenseNet

- `densenet_169`, `densenet_121`, `densenet_201`

EfficientNet

- `efficientnetb0`, `efficientnetb1`, `efficientnetb2`, 
`efficientnetb3`, `efficientnetb4`, `efficientnetb5`, `efficientnetb6`, `efficientnetb7`


And the default model configurations are specified [here](https://github.com/triagemd/keras-model-specs/blob/master/keras_model_specs/model_specs.json).

### Keras Trainer definition

These are the default options:

```
OPTIONS = {
        'model_spec': {'type': str},
        'output_logs_dir': {'type': str},
        'output_model_dir': {'type': str},
        'activation': {'type': str, 'default': 'softmax'},
        'batch_size': {'type': int, 'default': 1},
        'callback_list': {'type': list, 'default': []},
        'checkpoint_path': {'type': str, 'default': None},
        'class_weights': {'type': None, 'default': None},
        'custom_crop': {'type': bool, 'default': False},
        'custom_model': {'type': None, 'default': None},
        'decay': {'type': float, 'default': 0.0005},
        'dropout_rate': {'type': float, 'default': 0.0},
        'epochs': {'type': int, 'default': 1},
        'freeze_layers_list': {'type': None, 'default': None},
        'loss_function': {'type': str, 'default': 'categorical_crossentropy'},
        'include_top': {'type': bool, 'default': False},
        'input_shape': {'type': None, 'default': None},
        'iterator_mode': {'type': str, 'default': None},
        'loss_weights': {'type': None, 'default': None},
        'max_queue_size': {'type': int, 'default': 16},
        'metrics': {'type': list, 'default': ['accuracy']},
        'model_kwargs': {'type': dict, 'default': {}},
        'momentum': {'type': float, 'default': 0.9},
        'num_classes': {'type': int, 'default': None},
        'num_gpus': {'type': int, 'default': 0},
        'optimizer': {'type': None, 'default': None},
        'pooling': {'type': str, 'default': 'avg'},
        'random_crop_size': {'type': float, 'default': None},
        'regularization_function': {'type': None, 'default': None},
        'regularization_layers': {'type': None, 'default': None},
        'regularize_bias': {'type': bool, 'default': False},
        'save_training_options': {'type': bool, 'default': True},
        'sgd_lr': {'type': float, 'default': 0.01},
        'top_layers': {'type': None, 'default': None},
        'track_sensitivity': {'type': bool, 'default': False},
        'train_data_generator': {'type': None, 'default': None},
        'train_dataset_dataframe_path': {'type': str, 'default': None},
        'train_dataset_dir': {'type': str, 'default': None},
        'train_generator': {'type': None, 'default': None},
        'val_data_generator': {'type': None, 'default': None},
        'val_dataset_dataframe_path': {'type': str, 'default': None},
        'val_dataset_dir': {'type': str, 'default': None},
        'val_generator': {'type': None, 'default': None},
        'verbose': {'type': bool, 'default': False},
        'weights': {'type': str, 'default': 'imagenet'},
        'workers': {'type': int, 'default': 1},
    }
```

You will find a guide of use [here](https://github.com/triagemd/keras-trainer/blob/master/example.ipynb)
