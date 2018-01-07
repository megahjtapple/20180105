# -*- encoding: utf-8 -*-

import numpy as np

print("开始。")

print("测试 a = np.linspace(0, 4, 5)")
a = np.linspace(0, 4, 5)
print("a 的形状：" + str(a.shape))
print("a 的 type：" + str(type(a)))
print("a 的 Data：")
print(str(a) + "\n")

print("测试 a = np.linspace(0, 4, 5)[:,np.newaxis]")
a = np.linspace(0, 4, 5)[:,np.newaxis]
print("a 的形状：" + str(a.shape))
print("a 的 type：" + str(type(a)))
print("a 的 Data：")
print(str(a) + "\n")

print("测试 a = np.linspace(0, 4, 5)[np.newaxis:]")
a = np.linspace(0, 4, 5)[np.newaxis,:]
print("a 的形状：" + str(a.shape))
print("a 的 type：" + str(type(a)))
print("a 的 Data：")
print(str(a) + "\n")


print("b 是数组。和 a 不同。不能 np.newaxis。")
b = [0, 1, 2]
print("测试 b = [0, 1, 2]:")
print(str(b))
print("b 的 type：" + str(type(b)) + "\n")

b = [[0, 1, 2], [3, 4, 5]]
print("测试 b = [[0, 1, 2], [3, 4, 5]]:")
print(str(b))
print("b 的 type：" + str(type(b)) + "\n")

print("结束。")























