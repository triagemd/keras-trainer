import numpy as np
import tensorflow as tf

from keras_trainer.losses import entropy_penalty_loss


def test_confidence_penalty_loss():
    # Start Tensorflow session
    sess = tf.InteractiveSession()

    # dummy
    y_true = [[1.0, 0.0, 0.0, 0.0]]
    y_pred = [[0.25, 0.25, 0.25, 0.25]]

    loss = entropy_penalty_loss(y_true, y_pred).eval()
    # max entropy = ln(4)
    assert loss == np.log(0.25)

    y_pred = [[1.0, 0.0, 0.0, 0.0]]
    loss = entropy_penalty_loss(y_true, y_pred).eval()
    # There are values that are 0
    assert np.isnan(loss[0])

    y_pred = [[0.85, 0.05, 0.05, 0.05]]
    loss = entropy_penalty_loss(y_true, y_pred).eval()
    assert loss == (0.85 * np.log(0.85) + 0.05 * np.log(0.05) + 0.05 * np.log(0.05) + 0.05 * np.log(0.05))

    sess.close()
