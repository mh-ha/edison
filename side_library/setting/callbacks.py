from abc import ABC, abstractmethod


class Callback(ABC):
    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_validation_begin(self):
        pass

    def on_validation_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass