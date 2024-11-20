import numpy as np

class Adam():
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-7):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w = None
        self.v_w = None
        self.t = 0

    def update(self, weights, grad_w):
        if self.m_w is None:
            self.m_w = np.zeros_like(grad_w)
            self.v_w = np.zeros_like(grad_w)

        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * grad_w ** 2

        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)

        weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)

        return weights