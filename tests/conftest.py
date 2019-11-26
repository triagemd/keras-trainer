import os
import keras
import pytest


@pytest.fixture('session')
def train_catdog_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'train'))


@pytest.fixture('session')
def train_catdog_dataset_json_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'train_data.json'))


@pytest.fixture('session')
def val_catdog_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'val'))


@pytest.fixture('session')
def val_catdog_dataset_json_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'val_data.json'))


@pytest.fixture('function')
def simple_model():
    input_layer = keras.layers.Input(shape=(224, 224, 3))
    model = keras.layers.Conv2D(3, (3, 3))(input_layer)
    model = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.models.Model(input_layer, model)

    top_layers = [keras.layers.Dense(2, name='dense'),
                  keras.layers.Activation('softmax', name='act_softmax')]

    # Layer Assembling
    for i, layer in enumerate(top_layers):
        if i == 0:
            top_layers[i] = layer(model.output)
        else:
            top_layers[i] = layer(top_layers[i - 1])

    # Final Model (last item of self.top_layer contains all of them assembled)
    return keras.models.Model(model.input, top_layers[-1])
