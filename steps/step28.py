if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

# 勾配降下法によりrosenbrock関数の最小値を求める
lr = 0.001 # 学習率
iters = 10000 # イテレーション数

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)
