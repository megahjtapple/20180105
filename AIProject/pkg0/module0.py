import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#200 random range from -0.5 to 0.5 in one column.
#x_data is uniform distributed. Values are always the same.
x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
#noise are some random numbers. Values are different each time.
noise = np.random.normal(0, 0.02, x_data.shape)
#y_data = x_data^2 + noise
y_data = np.square(x_data) + noise

print(y_data)