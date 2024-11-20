import numpy as np

class RMSprop():
    def __init__(self, lr=0.01, gamma=0.9, eps=1e-7):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.cache_w = None

    def update(self, weights, grad_w):

        if self.cache_w is None:
            self.cache_w = np.zeros_like(grad_w)

        self.cache_w = self.gamma * self.cache_w + (1 - self.gamma) * grad_w ** 2
        weights -= self.lr * grad_w / (np.sqrt(self.cache_w) + self.eps)

        return weights