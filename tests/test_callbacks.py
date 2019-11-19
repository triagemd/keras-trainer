import os

from keras.preprocessing import image
from keras_model_specs import ModelSpec
from keras_trainer.callbacks import SensitivityCallback


def test_sensitivity_callback(val_catdog_dataset_path):
    model_spec = ModelSpec.get('custom',
                               preprocess_func='between_plus_minus_1',
                               target_size=[96, 96, 3])

    val_data_generator = image.ImageDataGenerator(
        preprocessing_function=model_spec.preprocess_input
    )

    output_model_dir = 'tmp'
    batch_size = 10

    val_gen = val_data_generator.flow_from_directory(
        val_catdog_dataset_path,
        batch_size=batch_size,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False
    )

    sensitivity_callback = SensitivityCallback(val_gen,
                                               output_model_dir=output_model_dir,
                                               batch_size=batch_size)

    assert sensitivity_callback.max_sensitivity == -1
    assert sensitivity_callback.batch_size == 10
    assert sensitivity_callback.output_model_dir == os.path.join(output_model_dir, 'model_max_sensitivity.hdf5')
