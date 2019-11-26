import os
import warnings
import numpy as np
import numpy.random as random
import multiprocessing.pool

from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras_preprocessing.image.iterator import BatchFromFilesMixin, Iterator
from keras_preprocessing.image.utils import validate_filename, _list_valid_filenames_in_directory

from keras_preprocessing import get_keras_submodule

backend = get_keras_submodule('backend')


# These functions are modifications on the original Keras functions, for the original documentation of the classes
# we refer the user to https://github.com/keras-team/keras-preprocessing/tree/master/keras_preprocessing/image


class EnhancedBatchFromFilesMixin(BatchFromFilesMixin):
    """Adds methods related to getting batches from filenames.
    It includes the logic to transform image files to batches.
    Addition of a method to random crop images.
    """

    def set_processing_attrs(self,
                             image_data_generator,
                             random_crop_size,
                             n_outputs,
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
            random_crop_size: Size of the random crop. Either a percentage of the original image (0,1) that will do square
                crop, a fixed size (tuple), or integer where the value will set equally to both dimensions.
            target_size: tuple of integers, dimensions to resize input images to.
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

        if isinstance(random_crop_size, int):
            self.random_crop_size = (int(random_crop_size), int(random_crop_size))
            self.crop_mode = 'size'
        elif isinstance(random_crop_size, float) and 0.0 < random_crop_size < 1.0:
            self.random_crop_size = random_crop_size
            self.crop_mode = 'percentage'
        else:
            self.random_crop_size = random_crop_size
            self.crop_mode = 'size'

        if not isinstance(n_outputs, int) or n_outputs < 1:
            raise ValueError('Incorrect value for n_outputs, must be an int equal or higher than 1')
        else:
            self.n_outputs = n_outputs

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
            if self.random_crop_size:
                img = img.crop(self.random_crop_parameters(img,
                                                           crop_size=self.random_crop_size,
                                                           crop_mode=self.crop_mode))
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
        elif self.class_mode == 'raw' or self.class_mode == 'probabilistic':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.n_outputs > 1:
            batch_y = [batch_y for n in range(0, self.n_outputs)]
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]


class EnhancedIterator(Iterator):
    def set_iterator_mode(self,
                          mode=None,
                          classes=None):
        """

        Args:
            mode:
            - None: Each sample is selected randomly.
            - 'equiprobable': Each sample is selected randomly with uniform class probability, so all the classes are
                                  evenly distributed. We can have repetition of samples during the same epoch.
            classes: Array where every index is the sample class. It belongs to the classes parameter of a Keras generator.

        """
        self.mode = mode

        if self.mode == 'equiprobable':
            if self.classes is not None:
                # Class array divides the 1D array into an index list per each class.
                # E.g. [0, 0, 1, 1] --> [array([0, 1]), array([3, 4)]
                self.num_classes = len(np.unique(classes))
                self.class_array = [np.where(i == classes)[0] for i in range(0, self.num_classes)]
            else:
                raise ValueError('`classes` variable cannot be None')

    def _set_index_array(self):
        if self.mode is None:
            self.index_array = np.arange(self.n)
            if self.shuffle:
                self.index_array = np.random.permutation(self.n)
        elif self.mode == 'equiprobable':
            self.index_array = np.array([np.random.choice(
                self.class_array[np.random.choice(self.num_classes, 1)[0]], 1)[0] for k in range(self.n)])


class EnhancedDirectoryIterator(EnhancedBatchFromFilesMixin, EnhancedIterator):
    """Iterator capable of reading images from a directory on disk, crop_size addition to perform image crops.
    # Arguments
        directory: string, path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        random_crop_size: Size of the random crop. Either a percentage of the original image (0,1) that will do square
        crop, a fixed size (tuple), or integer where the value will set equally to both dimensions.
        n_outputs: Integer indicating the number of outputs of the model. It will duplicate the labels. That is useful for
             multi-loss functions.
        iterator_mode: - None: Each sample is selected randomly.
                       - 'equiprobable': Each sample is selected randomly with uniform class probability, so all the
                     classes are evenly distributed. We can have repetition of samples during the same epoch.
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
                 iterator_mode=None,
                 random_crop_size=None,
                 n_outputs=1,
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
        super(EnhancedDirectoryIterator, self).set_processing_attrs(image_data_generator,
                                                                    random_crop_size,
                                                                    n_outputs,
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
        self.iterator_mode = iterator_mode

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
        # Init iterator
        super(EnhancedDirectoryIterator, self).__init__(self.samples,
                                                        batch_size,
                                                        shuffle,
                                                        seed)
        # Set mode
        super(EnhancedDirectoryIterator, self).set_iterator_mode(self.iterator_mode,
                                                                 self.classes)

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


class EnhancedDataFrameIterator(EnhancedBatchFromFilesMixin, EnhancedIterator):
    """Iterator capable of reading images from a directory on disk through a dataframe.
    Args:
        dataframe: Pandas dataframe containing the filepaths relative to
            `directory` (or absolute paths if `directory` is None) of the
            images in a string column. It should include other column/s
            depending on the `class_mode`:
            - if `class_mode` is `"categorical"` (default value) it must
                include the `y_col` column with the class/es of each image.
                Values in column can be string/list/tuple if a single class
                or list/tuple if multiple classes.
            - if `class_mode` is `"binary"` or `"sparse"` it must include
                the given `y_col` column with class values as strings.
            - if `class_mode` is `"raw"` or `"multi_output"` it should contain
                the columns specified in `y_col`.
            - if `class_mode` is `"input"` or `None` no extra column is needed.
        directory: string, path to the directory to read images from. If `None`,
            data in `x_col` column should be absolute paths.
        image_data_generator: Instance of `ImageDataGenerator` to use for
            random transformations and normalization. If None, no transformations
            and normalizations are made.
        n_outputs: Integer indicating the number of outputs of the model. It will duplicate the labels. That is useful for
             multi-loss functions.
        iterator_mode: - None: Each sample is selected randomly.
                       - 'equiprobable': Each sample is selected randomly with uniform class probability, so all the
                     classes are evenly distributed. We can have repetition of samples during the same epoch.
        x_col: string, column in `dataframe` that contains the filenames (or
            absolute paths if `directory` is `None`).
        y_col: string or list, column/s in `dataframe` that has the target data.
        weight_col: string, column in `dataframe` that contains the sample
            weights. Default: `None`.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, classes to use (e.g. `["dogs", "cats"]`).
            If None, all classes in `y_col` will be used.
        class_mode: one of "binary", "categorical", "input", "multi_output",
            "raw", "sparse" or None. Default: "categorical".
            Mode for yielding the targets:
            - `"binary"`: 1D numpy array of binary labels,
            - `"categorical"`: 2D numpy array of one-hot encoded labels.
                Supports multi-label output.
            - `"input"`: images identical to input images (mainly used to
                work with autoencoders),
            - `"multi_output"`: list with the values of the different columns,
            - `"raw"`: numpy array of values in `y_col` column(s),
            - `"sparse"`: 1D numpy array of integer labels,
            - `"probabilistic"`: List of probabilities
            - `None`, no targets are returned (the generator will only yield
                batches of image data, which is useful to use in
                `model.predict_generator()`).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
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
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for the generated arrays.
        validate_filenames: Boolean, whether to validate image filenames in
        `x_col`. If `True`, invalid images will be ignored. Disabling this option
        can lead to speed-up in the instantiation of this class. Default: `True`.
    """
    allowed_class_modes = {
        'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse', None, 'probabilistic'
    }

    def __init__(self,
                 dataframe,
                 image_data_generator=None,
                 random_crop_size=None,
                 directory=None,
                 n_outputs=1,
                 iterator_mode=None,
                 x_col="filename",
                 y_col="class",
                 weight_col=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='probabilistic',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 validate_filenames=True):

        super(EnhancedDataFrameIterator, self).set_processing_attrs(image_data_generator,
                                                                    random_crop_size,
                                                                    n_outputs,
                                                                    target_size,
                                                                    color_mode,
                                                                    data_format,
                                                                    save_to_dir,
                                                                    save_prefix,
                                                                    save_format,
                                                                    subset,
                                                                    interpolation)
        df = dataframe.copy()
        self.directory = directory or ''
        self.class_mode = class_mode
        self.dtype = dtype
        self.iterator_mode = iterator_mode
        # check that inputs match the required class_mode
        self._check_params(df, x_col, y_col, weight_col, classes)
        if validate_filenames:  # check which image files are valid and keep them
            df = self._filter_valid_filepaths(df, x_col)

        if class_mode == "probabilistic":
            # Probabilistic target array
            self._targets = np.array([value for value in df[y_col].values])
            # We assign only the class with more probabilty
            self.classes = np.array([np.argmax(target) for target in self._targets])
            num_classes = len(self._targets[0])
        if class_mode not in ["input", "multi_output", "raw", "probabilistic", None]:
            df, classes = self._filter_classes(df, y_col, classes)
            num_classes = len(classes)
            # build an index of all the unique classes
            self.class_indices = dict(zip(classes, range(len(classes))))
        # retrieve only training or validation set
        if self.split:
            num_files = len(df)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            df = df.iloc[start: stop, :]
        # get labels for each observation
        if class_mode not in ["input", "multi_output", "raw", "probabilistic", None]:
            self.classes = self.get_classes(df, y_col)
        self.filenames = df[x_col].tolist()
        self._sample_weight = df[weight_col].values if weight_col else None

        if class_mode == "multi_output":
            self._targets = [np.array(df[col].tolist()) for col in y_col]
        if class_mode == "raw":
            self._targets = df[y_col].values
        self.samples = len(self.filenames)
        validated_string = 'validated' if validate_filenames else 'non-validated'
        if class_mode in ["input", "multi_output", "raw", None]:
            print('Found {} {} image filenames.'
                  .format(self.samples, validated_string))
        else:
            print('Found {} {} image filenames belonging to {} classes.'
                  .format(self.samples, validated_string, num_classes))
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        # Init Iterator
        super(EnhancedDataFrameIterator, self).__init__(self.samples,
                                                        batch_size,
                                                        shuffle,
                                                        seed)
        # Set mode
        super(EnhancedDataFrameIterator, self).set_iterator_mode(self.iterator_mode,
                                                                 self.classes)

    def _check_params(self, df, x_col, y_col, weight_col, classes):
        # check class mode is one of the currently supported
        if self.class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(self.class_mode, self.allowed_class_modes))

        if self.class_mode == 'probabilistic':
            if not all(df[y_col].apply(lambda x: isinstance(x, list))):
                raise TypeError('All values in column y_col={} must be list.'
                                .format(x_col))
            class_number = len(df[y_col].values[0])
            if not all(df[y_col].apply(lambda x: len(x) == class_number)):
                raise TypeError('All lists in column y_col={} must have same length.'
                                .format(x_col))

        # check that y_col has several column names if class_mode is multi_output
        if (self.class_mode == 'multi_output') and not isinstance(y_col, list):
            raise TypeError(
                'If class_mode="{}", y_col must be a list. Received {}.'
                .format(self.class_mode, type(y_col).__name__)
            )
        # check that filenames/filepaths column values are all strings
        if not all(df[x_col].apply(lambda x: isinstance(x, str))):
            raise TypeError('All values in column x_col={} must be strings.'
                            .format(x_col))
        # check labels are string if class_mode is binary or sparse
        if self.class_mode in {'binary', 'sparse'}:
            if not all(df[y_col].apply(lambda x: isinstance(x, str))):
                raise TypeError('If class_mode="{}", y_col="{}" column '
                                'values must be strings.'
                                .format(self.class_mode, y_col))
        # check that if binary there are only 2 different classes
        if self.class_mode == 'binary':
            if classes:
                classes = set(classes)
                if len(classes) != 2:
                    raise ValueError('If class_mode="binary" there must be 2 '
                                     'classes. {} class/es were given.'
                                     .format(len(classes)))
            elif df[y_col].nunique() != 2:
                raise ValueError('If class_mode="binary" there must be 2 classes. '
                                 'Found {} classes.'.format(df[y_col].nunique()))
        # check values are string, list or tuple if class_mode is categorical
        if self.class_mode == 'categorical':
            types = (str, list, tuple)
            if not all(df[y_col].apply(lambda x: isinstance(x, types))):
                raise TypeError('If class_mode="{}", y_col="{}" column '
                                'values must be type string, list or tuple.'
                                .format(self.class_mode, y_col))
        # raise warning if classes are given but will be unused
        if classes and self.class_mode in {"input", "multi_output", "raw", None}:
            warnings.warn('`classes` will be ignored given the class_mode="{}"'
                          .format(self.class_mode))
        # check that if weight column that the values are numerical
        if weight_col and not issubclass(df[weight_col].dtype.type, np.number):
            raise TypeError('Column weight_col={} must be numeric.'
                            .format(weight_col))

    def get_classes(self, df, y_col):
        labels = []
        for label in df[y_col]:
            if isinstance(label, (list, tuple)):
                labels.append([self.class_indices[lbl] for lbl in label])
            else:
                labels.append(self.class_indices[label])
        return labels

    @staticmethod
    def _filter_classes(df, y_col, classes):
        df = df.copy()

        def remove_classes(labels, classes):
            if isinstance(labels, (list, tuple)):
                labels = [cls for cls in labels if cls in classes]
                return labels or None
            elif isinstance(labels, str):
                return labels if labels in classes else None
            else:
                raise TypeError(
                    "Expect string, list or tuple but found {} in {} column "
                    .format(type(labels), y_col)
                )

        if classes:
            classes = set(classes)  # sort and prepare for membership lookup
            df[y_col] = df[y_col].apply(lambda x: remove_classes(x, classes))
        else:
            classes = set()
            for v in df[y_col]:
                if isinstance(v, (list, tuple)):
                    classes.update(v)
                else:
                    classes.add(v)
        return df.dropna(subset=[y_col]), sorted(classes)

    def _filter_valid_filepaths(self, df, x_col):
        """Keep only dataframe rows with valid filenames
        Args:
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or filepaths
       Returns:
            absolute paths to image files
        """
        filepaths = df[x_col].map(
            lambda fname: os.path.join(self.directory, fname)
        )
        mask = filepaths.apply(validate_filename, args=(self.white_list_formats,))
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                'These filename(s) will be ignored.'
                .format(n_invalid, x_col)
            )
        return df[mask]

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        if self.class_mode in {"multi_output", "raw", "probabilistic"}:
            return self._targets
        else:
            return self.classes

    @property
    def sample_weight(self):
        return self._sample_weight


class EnhancedImageDataGenerator(ImageDataGenerator):
    """

    This is a modification of the Keras class ImageDataGenerator that includes some extra functionalities.
    The image data generator has a new option `random_crop_size` that if specified will perform random crops prior to
    resize the image to the specified target size.

    - random_crop_size:  Size of the random crop. Either a percentage of the original image (0,1) that will do square
        crop, a fixed size (tuple), or integer where the value will set equally to both dimensions.
        target_size: tuple of integers, dimensions to resize input images to.

    The functions `flow_from_dataframe` and `flow_from_directory` incorporate the following extra variables:

    - n_outputs: Integer indicating the number of outputs of the model. It will duplicate the labels. That is useful for
    multi-loss functions.
    - iterator_mode: - None: Each sample is selected randomly.
                     - 'equiprobable': Each sample is selected randomly with uniform class probability, so all the
                     classes are evenly distributed. We can have repetition of samples during the same epoch.


    The `flow_from_directory` function accepts class_mode='probabilistic' to handle a list of labels in a
    dataframe column.

    """

    def __init__(self,
                 random_crop_size=None,
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

        self.random_crop_size = random_crop_size

        super(EnhancedImageDataGenerator, self).__init__(
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

    def flow_from_dataframe(self,
                            dataframe,
                            directory=None,
                            n_outputs=1,
                            iterator_mode=None,
                            x_col="filename",
                            y_col="class",
                            weight_col=None,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='probabilistic',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            data_format='channels_last',
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            dtype='float32',
                            validate_filenames=True
                            ):

        return EnhancedDataFrameIterator(
            dataframe,
            iterator_mode=iterator_mode,
            random_crop_size=self.random_crop_size,
            n_outputs=n_outputs,
            directory=directory,
            image_data_generator=self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            dtype=dtype,
            validate_filenames=validate_filenames
        )

    def flow_from_directory(self,
                            directory,
                            iterator_mode=None,
                            n_outputs=1,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):

        return EnhancedDirectoryIterator(
            directory=directory,
            image_data_generator=self,
            random_crop_size=self.random_crop_size,
            n_outputs=n_outputs,
            iterator_mode=iterator_mode,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)
