import os
import json
import platform
import keras
import tensorflow
import copy
import numbers

from six import string_types
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import image
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.mobilenet import MobileNet
from keras_model_specs import ModelSpec
from keras_trainer.parallel import make_parallel


class Trainer(object):

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

    def __init__(self, **options):
        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key, ))
            value = options.get(key, copy.copy(option.get('default')))
            setattr(self, key, value)

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options), ))

        if isinstance(self.model_spec, string_types):
            self.model_spec = ModelSpec.get(self.model_spec)
        elif isinstance(self.model_spec, dict):
            self.model_spec = ModelSpec.get(self.model_spec['name'], **self.model_spec)

        if self.num_classes is None and self.top_layers is None:
            raise ValueError('num_classes must be set to use the default fully connected + softmax top_layers')

        options = dict([(key, getattr(self, key)) for key in self.OPTIONS.keys() if getattr(self, key) is not None])
        options['model_spec'] = self.model_spec.as_json()
        self.context = {
            'versions': {
                'python': platform.python_version(),
                'tensorflow': tensorflow.__version__,
                'keras': keras.__version__
            },
            'options': options
        }

        # Initialize the model instance
        if self.checkpoint_path is not None:
            self.model = load_model(self.checkpoint_path)
        else:
            if self.model_spec.klass == MobileNet:
                # Initialize the base with valid target_size
                model_kwargs = {
                    'input_shape': (224, 224, 3),
                    'weights': self.weights,
                    'include_top': self.include_top,
                    'pooling': self.pooling
                }
                model_kwargs.update(self.model_kwargs)
                self.model = self.model_spec.klass(**model_kwargs)

                # Expand the base model to match the spec target_size
                model_kwargs.update({
                    'input_shape': self.input_shape or self.model_spec.target_size,
                    'weights': None
                })
                expanded_model = self.model_spec.klass(**model_kwargs)
                for i in range(1, len(expanded_model.layers)):
                    expanded_model.layers[i].set_weights(self.model.layers[i].get_weights())
                self.model = expanded_model
            else:
                self.model = self.model_spec.klass(
                    input_shape=self.input_shape or self.model_spec.target_size,
                    weights=self.weights,
                    include_top=self.include_top,
                    pooling=self.pooling
                )

            # If top layers are given include them, else include a Dense Layer with Softmax/Sigmoid
            # (Dropout optional if dropout_rate entered as parameter)
            if self.top_layers is None:
                # Init list of layers
                self.top_layers = []

                # Include Dropout
                if self.dropout_rate > 0.0:
                    self.top_layers.append(Dropout(self.dropout_rate))

                # Set Dense Layer
                self.top_layers.append(Dense(self.num_classes, name='dense'))

                # Set Activation Layer
                if self.activation == 'sigmoid':
                    self.top_layers.append(Activation('sigmoid', name='act_sigmoid'))
                elif self.activation == 'softmax':
                    self.top_layers.append(Activation('softmax', name='act_softmax'))

            # Layer Assembling
            for i, layer in enumerate(self.top_layers):
                if i == 0:
                    self.top_layers[i] = layer(self.model.output)
                else:
                    self.top_layers[i] = layer(self.top_layers[i - 1])

            # Final Model (last item of self.top_layer contains all of them assembled)
            self.model = Model(self.model.input, self.top_layers[-1])

        # Freeze layers if contained in list
        if self.freeze_layers_list is not None:
            for layer in self.freeze_layers_list:
                if isinstance(layer, numbers.Number):
                    self.model.layers[layer].trainable = False
                elif isinstance(layer, str):
                    self.model.get_layer(layer).trainable = False

        # Print the model summary.
        if self.verbose:
            self.model.summary()

        # If num_gpus is higher than one, we parallelize the model
        if self.num_gpus > 1:
            self.model = make_parallel(self.model, self.num_gpus)

        # Override the optimizer or use the default.
        self.optimizer = self.optimizer or optimizers.SGD(
            lr=self.sgd_lr,
            decay=self.decay,
            momentum=self.momentum,
            nesterov=True
        )

        if not os.path.exists(self.output_model_dir):
            os.makedirs(self.output_model_dir)

        # To complement Keras message
        print('Training data')

        # Set up the training data generator.
        self.train_data_generator = self.train_data_generator or image.ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0,
            height_shift_range=0,
            preprocessing_function=self.model_spec.preprocess_input,
            shear_range=0,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        self.train_gen = self.train_generator or self.train_data_generator.flow_from_directory(
            self.train_dataset_dir,
            batch_size=self.batch_size,
            target_size=self.model_spec.target_size[:2],
            class_mode='categorical'
        )

        # To complement Keras message
        print('Validation data')

        # Set up the validation data generator.
        self.val_data_generator = self.val_data_generator or image.ImageDataGenerator(
            preprocessing_function=self.model_spec.preprocess_input
        )

        self.val_gen = self.val_generator or self.val_data_generator.flow_from_directory(
            self.val_dataset_dir,
            batch_size=self.batch_size,
            target_size=self.model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False
        )

    def run(self):
        # Set Checkpoint to save Highest Accuracy Model
        checkpoint_acc = ModelCheckpoint(
            os.path.join(self.output_model_dir, 'model_max_acc.hdf5'),
            verbose=1,
            monitor='val_acc',
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
        self.callback_list.append(checkpoint_acc)

        # Set Checkpoint to save Minimum Loss Model
        checkpoint_loss = ModelCheckpoint(
            os.path.join(self.output_model_dir, 'model_min_loss.hdf5'),
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min'
        )
        self.callback_list.append(checkpoint_loss)

        # Set Tensorboard Visualization
        tensorboard = TensorBoard(
            log_dir=self.output_logs_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )
        tensorboard.set_model(self.model)
        self.callback_list.append(tensorboard)

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        # Model training
        self.history = self.model.fit_generator(
            self.train_gen,
            verbose=1,
            steps_per_epoch=self.train_gen.samples // self.batch_size,
            epochs=self.epochs,
            callbacks=self.callback_list,
            validation_data=self.val_gen,
            validation_steps=self.val_gen.samples // self.batch_size,
            workers=self.workers,
            class_weight=self.class_weights,
            max_queue_size=self.max_queue_size
        )

        # Save last model
        self.model.save(os.path.join(self.output_model_dir, 'final_model.hdf5'))

        # Dump model_spec.json file
        with open(os.path.join(self.output_model_dir, 'model_spec.json'), 'w') as file:
            file.write(json.dumps(self.model_spec.as_json(), indent=True, sort_keys=True))

        # Save training options
        with open(os.path.join(self.output_model_dir, 'training_options.json'), 'w') as file:
            safe_options = {}
            for key, value in self.context['options'].items():
                if value is None:
                    continue
                try:
                    json.dumps(value)
                    safe_options[key] = value
                except TypeError:
                    continue
            self.context['options'] = safe_options
            file.write(json.dumps(self.context, indent=True, sort_keys=True))
