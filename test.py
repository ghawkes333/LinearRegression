import numpy as np

params = np.array([[1, 2], [3, 4]])
input = np.array([[10, 100], [1000, 10000]])

print(params * input)
print(input * params)
print(np.sum(params*input))