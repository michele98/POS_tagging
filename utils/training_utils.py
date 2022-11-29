import os
import numpy as np
from keras.callbacks import History


class MyHistory(History):
    """Load training history from previous trainings and save the history at the end of each epoch.
    Pass the folder in which the history must be saved. The same folder in which the weights are saved is fine.
    """
    def __init__(self, history_path = "./history.npy"):
        super().__init__()
        self.history_path = history_path
        if os.path.exists(history_path):
            self.history = np.load(history_path, allow_pickle='TRUE').item()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        np.save(self.history_path, self.history)
