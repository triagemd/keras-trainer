import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator, get_keras_submodule, array_to_img, \
    load_img, img_to_array

backend = get_keras_submodule('backend')


# These functions are modifications on the original Keras functions, for the documentation of the classes we refer the
# user to https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py


class BalancedDirectoryIterator(DirectoryIterator):
    '''
    Each sample is selected randomly and with uniform probability, so all the classes are distributed equiprobably.
    We can have repetition of samples during the same epoch.
    '''

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):
        super(BalancedDirectoryIterator, self).__init__(directory,
                                                        image_data_generator,
                                                        target_size,
                                                        color_mode,
                                                        classes,
                                                        class_mode,
                                                        batch_size,
                                                        shuffle,
                                                        seed,
                                                        data_format,
                                                        save_to_dir,
                                                        save_prefix,
                                                        save_format,
                                                        follow_links,
                                                        subset,
                                                        interpolation)

        self.class_array = [np.where(i == self.classes)[0] for i in range(0, self.num_classes)]

    def _set_index_array(self):
        self.index_array = np.array(
            [np.random.choice(self.class_array[np.random.choice(self.num_classes, 1)[0]], 1)[0]
             for k in range(0, self.samples)])


class BalancedImageDataGenerator(ImageDataGenerator):
    '''
    ImageDataGenerator that returns a balanced number of samples using a BalancedDirectoryIterator as iterator
    '''

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0,
                 ):

        super(BalancedImageDataGenerator, self).__init__(featurewise_center=featurewise_center,
                                                         samplewise_center=samplewise_center,
                                                         featurewise_std_normalization=featurewise_std_normalization,
                                                         samplewise_std_normalization=samplewise_std_normalization,
                                                         zca_whitening=zca_whitening,
                                                         zca_epsilon=zca_epsilon,
                                                         rotation_range=rotation_range,
                                                         width_shift_range=width_shift_range,
                                                         height_shift_range=height_shift_range,
                                                         brightness_range=brightness_range,
                                                         shear_range=shear_range,
                                                         zoom_range=zoom_range,
                                                         channel_shift_range=channel_shift_range,
                                                         fill_mode=fill_mode,
                                                         cval=cval,
                                                         horizontal_flip=horizontal_flip,
                                                         vertical_flip=vertical_flip,
                                                         rescale=rescale,
                                                         preprocessing_function=preprocessing_function,
                                                         data_format=data_format,
                                                         validation_split=validation_split)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return BalancedDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


class DirectoryIteratorSameMultiGT(DirectoryIterator):
    '''
    This iterator returns batch_x and a list containing n_outputs times batch_y, which are the ground truth labels.
    '''

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 n_outputs=2):
        super(DirectoryIteratorSameMultiGT, self).__init__(directory,
                                                           image_data_generator,
                                                           target_size,
                                                           color_mode,
                                                           classes,
                                                           class_mode,
                                                           batch_size,
                                                           shuffle,
                                                           seed,
                                                           data_format,
                                                           save_to_dir,
                                                           save_prefix,
                                                           save_format,
                                                           follow_links,
                                                           subset,
                                                           interpolation
                                                           )
        if not isinstance(n_outputs, int) or n_outputs < 1:
            raise ValueError('Incorrect value for n_outputs, must be an int equal or higher than 1')
        else:
            self.n_outputs = n_outputs

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=backend.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, [batch_y for n in range(0, self.n_outputs)]


class ImageDataGeneratorSameMultiGT(ImageDataGenerator):
    '''
    ImageDataGenerator that returns multiple outputs using DirectoryIteratorSameMultiGT as iterator
    '''

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0,
                 n_outputs=2):

        super(ImageDataGeneratorSameMultiGT, self).__init__(featurewise_center=featurewise_center,
                                                            samplewise_center=samplewise_center,
                                                            featurewise_std_normalization=featurewise_std_normalization,
                                                            samplewise_std_normalization=samplewise_std_normalization,
                                                            zca_whitening=zca_whitening,
                                                            zca_epsilon=zca_epsilon,
                                                            rotation_range=rotation_range,
                                                            width_shift_range=width_shift_range,
                                                            height_shift_range=height_shift_range,
                                                            brightness_range=brightness_range,
                                                            shear_range=shear_range,
                                                            zoom_range=zoom_range,
                                                            channel_shift_range=channel_shift_range,
                                                            fill_mode=fill_mode,
                                                            cval=cval,
                                                            horizontal_flip=horizontal_flip,
                                                            vertical_flip=vertical_flip,
                                                            rescale=rescale,
                                                            preprocessing_function=preprocessing_function,
                                                            data_format=data_format,
                                                            validation_split=validation_split)

        self.n_outputs = n_outputs

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest',
                            ):

        return DirectoryIteratorSameMultiGT(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            n_outputs=self.n_outputs
        )
