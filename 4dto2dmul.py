import numpy as np
X, Y, Z, W = 7, 9, 2, 4
A = np.random.rand(X, Y, Z, W)
B = np.random.rand(X, Y, W, Z)

# 4-dimensional matrix multiplication using 2D matmul operation 

# Option 1
A_3d = A.reshape(X*Y, Z, W)
B_3d = B.reshape(X*Y, W, Z)

A_extended = np.zeros((X*Y * Z, X*Y * W))
for i in range(X*Y):
    A_extended[i*Z: i*Z+Z, i*W:i*W+W] = A_3d[i, :, :]

B_extended = B_3d.reshape(X*Y * W, Z)
result_2d = A_extended @ B_extended
result_3d = result_2d.reshape(X*Y, Z, Z)
# Reshape result back into the original shape
result = result_3d.reshape(X, Y, Z, Z)

# Option 2
# # Initialize a result matrix with the same shape
# result = np.zeros((X, Y, Z, Z))

# # Iterate over the first two dimensions
# for i in range(X):
#     for j in range(Y):
#         # Perform 2D matrix multiplication for each pair of 2D matrices in the last two dimensions
#         result[i, j] = A[i, j] @ B[i, j]
#

# Reshape A and B into 2D matrices

# Uncomment the following line to make sure the asertion works as expected
# result[0][0][0][0]=0
np.testing.assert_array_equal(result, A@B)
