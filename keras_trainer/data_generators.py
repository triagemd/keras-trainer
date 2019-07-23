import os
import numpy as np
import numpy.random as random
import multiprocessing.pool

from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator, array_to_img, load_img, img_to_array
from keras_preprocessing.image.iterator import BatchFromFilesMixin, Iterator
from keras_preprocessing.image.utils import _list_valid_filenames_in_directory

from keras_preprocessing import get_keras_submodule

backend = get_keras_submodule('backend')


# These functions are modifications on the original Keras functions, for the documentation of the classes we refer the
# user to https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py


class BalancedDirectoryIterator(DirectoryIterator):
    '''

    This iterator inherits from DirectoryIterator
    (https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py#L1811)
    Each sample is selected randomly with uniform probability, so all the classes are evenly distributed.
    We can have repetition of samples during the same epoch.

    '''

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
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

    ImageDataGenerator that returns a balanced number of samples using a BalancedDirectoryIterator as iterator.

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
                 data_format='channels_last',
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

    This iterator inherits from DirectoryIterator
    (https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py#L1811)
    It has the addition of `n_outputs` as an argument.

    Returns `batch_x` and a list containing `n_outputs` times `batch_y`, which are the ground truth labels.

    '''

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
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

    ImageDataGenerator that returns multiple outputs using DirectoryIteratorSameMultiGT as iterator.

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
                 data_format='channels_last',
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


class BatchFromFilesMixinMod(BatchFromFilesMixin):
    """Adds methods related to getting batches from filenames
    It includes the logic to transform image files to batches.
    Addition of a method to random crop images.
    """

    def set_processing_attrs(self,
                             image_data_generator,
                             crop_size,
                             target_size,
                             color_mode,
                             data_format,
                             save_to_dir,
                             save_prefix,
                             save_format,
                             subset,
                             interpolation):
        """Sets attributes to use later for processing files into a batch.
        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

        if isinstance(crop_size, int):
            self.crop_size = (int(crop_size), int(crop_size))
            self.crop_mode = 'size'
        elif isinstance(crop_size, float) and 0.0 < crop_size < 1.0:
            self.crop_size = crop_size
            self.crop_mode = 'percentage'
        else:
            self.crop_size = crop_size
            self.crop_mode = 'size'

    @staticmethod
    def random_crop_parameters(img, crop_size, crop_mode='size'):
        w, h = img.size

        if crop_mode == 'percentage':
            size_crop = int(crop_size * min(img.size))
            tw, th = size_crop, size_crop
        else:
            tw, th = crop_size

        if w <= tw or h <= th:
            return 0, 0, w, h

        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)

        return x, y, x + tw, y + th

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=None,
                           interpolation=self.interpolation)
            if self.crop_size:
                img = img.crop(self.random_crop_parameters(img, crop_size=self.crop_size, crop_mode=self.crop_mode))
            img = img.resize(self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
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
                print(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]


class RandomCropDirectoryIterator(BatchFromFilesMixinMod, Iterator):
    """Iterator capable of reading images from a directory on disk, crop_size addition to perform image crops.
    # Arguments
        directory: string, path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        crop_size: Size of the random crop. Either a percentage of the original image (0,1) that will do square crop
            or a fixed size (tuple) or integer where integer will set both dimensions as equal.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        follow_links: boolean,follow symbolic links to subdirectories
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for generated arrays.
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 image_data_generator,
                 crop_size=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(RandomCropDirectoryIterator, self).set_processing_attrs(image_data_generator,
                                                                      crop_size,
                                                                      target_size,
                                                                      color_mode,
                                                                      data_format,
                                                                      save_to_dir,
                                                                      save_prefix,
                                                                      save_format,
                                                                      subset,
                                                                      interpolation)
        self.directory = directory
        self.classes = classes

        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, self.split,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super(RandomCropDirectoryIterator, self).__init__(self.samples,
                                                          batch_size,
                                                          shuffle,
                                                          seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None


class RandomCropImageDataGenerator(ImageDataGenerator):
    '''

    ImageDataGenerator that has can perform random crops if specified in flow_from_directory function

    '''

    def __init__(self,
                 crop_size=None,
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
                 data_format='channels_last',
                 validation_split=0.0,
                 ):

        super(RandomCropImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
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
        self.crop_size = crop_size

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

        return RandomCropDirectoryIterator(
            directory=directory, image_data_generator=self, crop_size=self.crop_size,
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
