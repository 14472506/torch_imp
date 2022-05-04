import numpy as np

f = np.array([2.01, 2.05, 2.11, 1.99, 2.05, 1.98], dtype=float)
grad = np.gradient(f, 3)
print(grad)