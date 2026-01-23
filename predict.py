import matplotlib.pyplot as plt
import numpy as np

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

plt.scatter(x_train, y_train)
plt.show()

plt.scatter(x_test, y_train)
plt.show()

