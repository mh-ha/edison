from .abstracts import Wrapper



# Define the PyTorch specific wrapper class
class PyTorchWrapper(Wrapper):
    def train(self, proof_ready):
        # PyTorch specific training logic
        pass

    def validation(self, proof_ready):
        # PyTorch specific validation logic
        pass