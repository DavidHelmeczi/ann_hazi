import numpy as np

class SGD_momentum():
    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.weights = None

    def update(self, weights, grad_w):
        if self.velocity is None:
            self.velocity = np.zeros_like(grad_w)

        self.velocity = self.momentum * self.velocity - self.learning_rate * grad_w
        weights += self.velocity
        
        return weights


