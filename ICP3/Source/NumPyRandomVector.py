import numpy as np

# Create random vector of size 15 having only Integers in the range 1-20
s = np.random.randint(1,20,15)
print(s)
print('\n')

# Reshape the array to 3 by 5
b = s.reshape((3,5))
print(b)
print('\n')

#b[np.arange(len(b)), b.argmax(1)] = 0
rowMaxe = b.max(axis=1).reshape(-1, 1)
print(rowMaxe.shape)
b[:] = np.where(b == rowMaxe, 0, b[:] )

print(b)