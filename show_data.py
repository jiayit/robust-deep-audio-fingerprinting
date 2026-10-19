import numpy as np

data = np.load("data/medeval_ids.npy")

print("Shape: ", data.shape)

print("Data type: ", data.dtype)

for i in range(20):
    print(f" {i}: ", data[i])

print("Sample: ", data[0])


