# 나쁜 구현
def bad_numerical_diff(f, x):
    h = 1e-50
    return (f(x+h) - f(x)) / h

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()
#
# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))

# 변수 2개인 경우
def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    # print(grad)
    return grad

# numerical_gradient(function_2, np.array[3.0, 4.0])

def gradiend_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    print(x)
    return x

init_x = np.array([-3.0, 4.0])
gradiend_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
