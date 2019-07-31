import numpy as np


from keras.regularizers import l2
from keras.applications import mobilenet
from keras_trainer.regularizations import set_model_regularization
from keras.models import Model
from keras.layers import SeparableConv2D, Conv2D, Dense, BatchNormalization, Activation


def test_set_model_regularization():
    model = mobilenet.MobileNet(alpha=0.25, include_top=False, pooling='avg', input_shape=[224, 224, 3])
    top_layers = []
    # Set Dense Layer
    top_layers.append(Dense(2, name='dense'))
    # Set Activation Layer
    top_layers.append(Activation('softmax', name='act_softmax'))

    # Layer Assembling
    for i, layer in enumerate(top_layers):
        if i == 0:
            top_layers[i] = layer(model.output)
        else:
            top_layers[i] = layer(top_layers[i - 1])

    # Final Model (last item of self.top_layer contains all of them assembled)
    model = Model(model.input, [top_layers[-1], top_layers[-1]])

    regularization_function = l2(0.00025)

    weigths_layer_expected = model.layers[1].get_weights()

    model = set_model_regularization(model, regularization_function)

    weigths_layer_actual = model.layers[1].get_weights()

    np.testing.assert_array_almost_equal(weigths_layer_actual, weigths_layer_expected)

    for layer in model.layers:
        if type(layer) in [SeparableConv2D, Conv2D, Dense, BatchNormalization]:
            assert len(layer.losses) > 0
