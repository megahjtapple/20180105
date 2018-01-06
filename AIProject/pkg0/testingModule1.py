# -*- encoding: utf-8 -*-

import numpy as np

print("开始。")

a = np.linspace(0, 1, 5)
print("a 的形状：" + str(a.shape))
print("a 的 Data：")
print(str(a) + "\n")

a = np.linspace(0, 1, 5)[:,np.newaxis]
print("a 的形状：" + str(a.shape))
print("a 的 Data：")
print(str(a) + "\n")

a = np.linspace(0, 1, 5)[np.newaxis:]
print("a 的形状：" + str(a.shape))
print("a 的 Data：")
print(str(a) + "\n")

print("结束。")
