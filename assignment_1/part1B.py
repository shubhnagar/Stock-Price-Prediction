import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(45)
  U = np.random.randn(N, d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))
  return U, M1, M2

def solve(N, d):
  U, M1, M2 = initialise_input(N, d)

  '''
  Enter your code here for steps 1 to 6
  '''
  # 1st
  X = np.matmul(U, M1)
  Y = np.matmul(U, M2)

  # 2nd
  row_indices = np.arange(1, N + 1).reshape(N, 1)
  X_cap = X + row_indices 

  # 3rd
  Z = np.matmul(X_cap, Y.T)
  rows, cols = np.indices(Z.shape)
  Z_sparsed = np.array(np.where((rows + cols) % 2 == 0, Z, 0))
  

  # 4th
  max_Z = np.max(Z_sparsed, axis=1, keepdims=True)  
  Z_s = Z_sparsed - max_Z 
  
  exp_Z = np.exp(Z_s)
  sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
  print(sum_exp_Z)
  Z_hat = exp_Z / sum_exp_Z
  print(Z_hat)
  max_indices = np.argmax(Z_hat, axis=1)
  return max_indices
  
N = 3
d = 3
max_indices = solve(N,d)
print(max_indices)
