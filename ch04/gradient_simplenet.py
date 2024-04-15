import sys, os


sys.path.append(os.pardir)
import numpy as np
from ch04.common.functions import softmax, cross_entropy_error
from ch04.common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = SimpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0, 0, 1])
loss = net.loss(x, t)
print(loss)

f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(f"dW: {dW}")