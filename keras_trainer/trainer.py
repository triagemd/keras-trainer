import os

from six import string_types
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense, Activation
from keras.preprocessing import image
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.mobilenet import MobileNet
from keras_model_specs import ModelSpec
from keras_trainer.parallel import make_parallel, detect_num_gpus


class Trainer(object):

    OPTIONS = {
        'model_spec': {'type': str},
        'train_dataset_dir': {'type': str},
        'val_dataset_dir': {'type': str},
        'output_model_dir': {'type': str},
        'output_logs_dir': {'type': str},
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
        'metrics': {'type': list, 'default': ['accuracy', ]},
        'batch_size': {'type': int, 'default': 1},
        'epochs': {'type': int, 'default': 1},
        'decay': {'type': float, 'default': 0.0005},
        'momentum': {'type': float, 'default': 0.9},
        'sgd_lr': {'type': float, 'default': 0.01},
        'pooling': {'type': str, 'default': 'avg'},
        'weights': {'type': str, 'default': 'imagenet'},
        'num_gpus': {'type': int, 'default': 1},
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
            value = options.get(key, option.get('default'))
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

        if self.num_gpus is None:
            self.num_gpus = detect_num_gpus()

    def run(self):
        # Set up the training data generator.
        train_data_generator = self.train_generator or image.ImageDataGenerator(
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

        train_gen = self.train_generator or train_data_generator.flow_from_directory(
            self.train_dataset_dir,
            batch_size=self.batch_size,
            target_size=self.model_spec.target_size[:2],
            class_mode='categorical'
        )

        # Set up the validation data generator.
        val_data_generator = self.val_data_generator or image.ImageDataGenerator(
            preprocessing_function=self.model_spec.preprocess_input
        )

        val_gen = self.val_generator or val_data_generator.flow_from_directory(
            self.val_dataset_dir,
            batch_size=self.batch_size,
            target_size=self.model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False
        )

        # Initialize the model instance.
        if self.checkpoint_path is not None:
            model = load_model(self.checkpoint_path)
        else:
            if self.model_spec.klass == MobileNet:
                # Initialize the base with valid target_size.
                model_kwargs = {
                    'input_shape': (224, 224, 3),
                    'weights': self.weights,
                    'include_top': False,
                    'pooling': self.pooling
                }
                model_kwargs.update(self.model_kwargs)
                model = self.model_spec.klass(**model_kwargs)

                # Expand the base model to match the spec target_size.
                model_kwargs.update({
                    'input_shape': self.model_spec.target_size,
                    'weights': None
                })
                expanded_model = self.model_spec.klass(**model_kwargs)
                for i in range(1, len(expanded_model.layers)):
                    expanded_model.layers[i].set_weights(model.layers[i].get_weights())
                model = expanded_model
            else:
                model = self.model_spec.klass(
                    input_shape=self.model_spec.target_size,
                    weights=self.weights,
                    include_top=False,
                    pooling=self.pooling
                )

            # Include the given top layers.
            if self.top_layers is None:
                layer = Dense(self.num_classes, name='dense')(model.output)
                self.top_layers = Activation('softmax', name='act_softmax')(layer)
            model = Model(model.input, self.top_layers)

        # Print the model summary.
        if self.verbose:
            model.summary()

        # GPU multiprocessing (if num_gpus is None we use all available GPUs)
        if self.num_gpus > 1:
            model = make_parallel(model, self.num_gpus)

        # Override the optimizer or use the default.
        optimizer = self.optimizer or optimizers.SGD(
            lr=self.sgd_lr,
            decay=self.decay,
            momentum=self.momentum,
            nesterov=True
        )

        if not os.path.exists(self.output_model_dir):
            os.makedirs(self.output_model_dir)

        checkpoint = ModelCheckpoint(
            os.path.join(self.output_model_dir, 'best.hdf5'),
            verbose=1,
            monitor='val_acc',
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )

        self.callback_list.append(checkpoint)

        tensorboard = TensorBoard(
            log_dir=self.output_logs_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )
        tensorboard.set_model(model)
        self.callback_list.append(tensorboard)

        model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        model.fit_generator(
            train_gen,
            verbose=1,
            steps_per_epoch=train_gen.samples // self.batch_size,
            epochs=self.epochs,
            callbacks=self.callback_list,
            validation_data=val_gen,
            validation_steps=val_gen.samples // self.batch_size,
            workers=self.workers,
            class_weight=self.class_weights,
            max_queue_size=self.max_queue_size
        )

        model.save(os.path.join(self.output_model_dir, 'final.hdf5'))
