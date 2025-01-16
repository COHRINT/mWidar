import numpy as np

# Create more matricies to test 
a = np.array([[1,2,3],[4,5,3]], dtype=float)

Abin = a.tobytes();

with open('example_2x3.bin','wb') as f:
    f.write(Abin)
