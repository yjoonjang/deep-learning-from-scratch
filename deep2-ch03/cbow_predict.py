import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

c0 = np.array([[1, 0, 0, 0, 0, 0, 0]]) # you
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]]) # goodbye

W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h = in_layer0.forward(c0) + in_layer1.forward(c1)
h = 0.5 * h

s = out_layer.forward(h)

print(s)
