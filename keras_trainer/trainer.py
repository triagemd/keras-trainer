import os

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Activation
from keras.preprocessing import image
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_model_specs import ModelSpec


class Trainer(object):

    OPTIONS = {
        'batch_size': {'type': int, 'default': 1},
        'epochs': {'type': int, 'default': 1},
        'alpha': {'type': int, 'default': 1},
        'decay': {'type': float, 'default': 0.0005},
        'momentum': {'type': float, 'default': 0.9},
        'sgd_lr': {'type': float, 'default': 0.01},
        'pooling': {'type': str, 'default': 'avg'},
        'weights': {'type': str, 'default': 'imagenet'},
        'workers': {'type': int, 'default': 1},
        'max_queue_size': {'type': int, 'default': 16}
    }

    def __init__(self, model_spec, dictionary, train_dataset_dir, val_dataset_dir, output_model_dir, output_logs_dir, **options):
        self.model_spec = ModelSpec.get(model_spec)
        self.dictionary = dictionary
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.output_model_dir = output_model_dir
        self.output_logs_dir = output_logs_dir
        self.train_generator = options.pop('train_generator', None)
        self.val_generator = options.pop('val_generator', None)
        self.top_layers = options.pop('top_layers', None)
        self.optimizer = options.pop('optimizer', None)
        self.callback_list = options.pop('callback_list', [])
        self.class_weights = options.pop('class_weights', None)
        self.loss_function = options.pop('loss_function', 'categorical_crossentropy')
        self.metrics = options.pop('metrics', ['accuracy'])

        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option %s' % (key, ))
            value = options.get(key, option.get('default'))
            setattr(self, key, value)

    def run(self):
        train_generator = self.train_generator or image.ImageDataGenerator(
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

        val_generator = self.val_generator or image.ImageDataGenerator(
            preprocessing_function=self.model_spec.preprocess_input
        )

        train_gen = train_generator.flow_from_directory(
            self.train_dataset_dir,
            batch_size=self.batch_size,
            target_size=self.model_spec.target_size[:2],
            class_mode='categorical'
        )

        val_gen = val_generator.flow_from_directory(
            self.val_dataset_dir,
            batch_size=self.batch_size,
            target_size=self.model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False
        )

        model = self.model_spec.klass(
            self.model_spec.target_size,
            alpha=self.alpha,
            weights=self.weights,
            include_top=False,
            pooling=self.pooling
        )

        if self.top_layers is None:
            layer = Dense(len(self.dictionary), name='dense')(model.output)
            self.top_layers = Activation('softmax', name='act_softmax')(layer)
        model = Model(model.input, self.top_layers)

        model.summary()

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
