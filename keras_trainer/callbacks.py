import os
import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import recall_score


class SensitivityCallback(Callback):
    def __init__(self, validation_data_generator, output_model_dir, batch_size=10):
        super(SensitivityCallback, self).__init__()
        self.validation_data_generator = validation_data_generator
        self.batch_size = batch_size
        self.output_model_dir = os.path.join(output_model_dir, 'model_max_sensitivity.hdf5')
        self.max_sensitivity = -1

    def on_epoch_end(self, epoch, logs={}):
        n_samples = self.validation_data_generator.samples
        n_batches = n_samples // self.batch_size
        last_batch_remainder = n_samples % self.batch_size

        y_true = []
        y_pred = []

        # Forward pass and store labels and predictions
        for i, batch in enumerate(self.validation_data_generator):
            if i == n_batches and last_batch_remainder > 0:
                batch = batch[0][0:last_batch_remainder], batch[1][0:last_batch_remainder]

            elif i > n_batches:
                break

            y_true += np.argmax(batch[1], axis=1).tolist()
            y_pred += np.argmax(self.model.predict(batch[0]), axis=1).tolist()

        # Compute the (unweighted) average sensitivity and save model if is better than previous epochs
        sensitivity = recall_score(y_true, y_pred, average='macro')
        logs['validation_sensitivity'] = sensitivity
        print('Epoch {epoch}, sensitivity={sensitivity}'.format(epoch=epoch, sensitivity=sensitivity))

        if sensitivity > self.max_sensitivity:
            print('Sensitivity increased from {max_sensitivity:.4f} to {sensitivity:.4f}, '
                  'saving model to {output_model_dir}'.format(
                      max_sensitivity=self.max_sensitivity,
                      sensitivity=sensitivity,
                      output_model_dir=self.output_model_dir))
            self.max_sensitivity = sensitivity
            self.model.save(self.output_model_dir)

        return
