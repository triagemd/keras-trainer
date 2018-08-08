import tensorflow as tf


def entropy_penalty_loss(y_true, y_pred):
    '''

    Args:
        y_true: Ground truth (Unused but needed for Keras formatting)
        y_pred: Network predicted probabilities

    Returns: The negative of the Entropy of the probability distribution (Note that is computed with natural log)

    '''

    return tf.reduce_sum(tf.multiply(y_pred, (tf.log(y_pred))), axis=1)
