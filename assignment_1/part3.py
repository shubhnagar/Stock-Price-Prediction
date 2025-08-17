import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
  def __init__(self):
    '''
    Initializing the parameters of the model

    Returns:
      None
    '''
    self.weights = None

  def fit(self, X, y):
    '''
    This function is used to obtain the weights of the model using closed form solution.

    Args:
      X : 2D numpy array of training set data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values in the training dataset. Dimensions (n x 1)

    Returns :
      None
    '''
    # Calculate the weights

    X_T = X.T
    X_d = np.matmul(X_T, X)
    

    if np.linalg.det(X_d) == 0:
       raise ValueError("The matrix is singular and cannot be inverted.")
       
    X_d_inv = np.linalg.inv(X_d)
    X_dd = np.matmul(X_d_inv, X_T)
    self.weights = np.matmul(X_dd, y)

    return
    raise NotImplementedError()

  def predict(self, X):
    '''
    This function is used to predict the target values for the given set of feature values

    Args:
      X: 2D numpy array of data points. Dimensions (n x (d+1))

    Returns:
      2D numpy array of predicted target values. Dimensions (n x 1)
    '''
    # Write your code here

    return np.matmul(X, self.weights)
  
  def regularize(self, X_validation, y_validation):
      n, m = X_validation.shape()
      I = np.eye(m)  # Identity matrix
      I[0, 0] = 0
      lamb = 0.5
      self.weights = np.linalg.inv(X_validation.T @ X_validation + lamb * I) @ X_validation.T @ y_validation


def transform_input(x):
    '''
    This function transforms the input to generate new features.

    Args:
      x: 2D numpy array of input values. Dimensions (n' x 1)

    Returns:
      2D numpy array of transformed input. Dimensions (n' x K+1)
      
    '''
    X_transformed = np.ones((x.shape[0], 4))
    X_transformed[:, 1] = x.flatten()
    X_transformed[:, 2] = x.flatten() ** 2
    X_transformed[:, 3] = np.cos(x.flatten())

    return X_transformed



    raise NotImplementedError()
    
def read_dataset(filepath):
    '''
    This function reads the dataset and creates train and test splits.
    
    n = 500
    n' = 0.9*n

    Args:
      filename: string containing the path of the csv file

    Returns:
      X_train: 2D numpy array of input values for training. Dimensions (n' x 1)
      y_train: 2D numpy array of target values for training. Dimensions (n' x 1)
      
      X_test: 2D numpy array of input values for testing. Dimensions ((n-n') x 1)
      y_test: 2D numpy array of target values for testing. Dimensions ((n-n') x 1)
      
    '''
    # Write your code here
    if "train.csv" in filepath:
      dataset = pd.read_csv(filepath)
      split_index =  int(len(dataset) * 0.8)
      x = dataset.drop(['ID', 'score'], axis=1).values
      y = dataset['score'].values

      # x_train = x[:split_index]
      # y_train = y[:split_index]


      # x_test = x[split_index:]
      # y_test = y[split_index:]
      # y_test = np.array(dataset['y'][split_index:][1:].values.reshape(-1, 64))
      # return x_train, y_train, x_test, y_test
      return x, y
    
    if "test.csv" in filepath:
      dataset = pd.read_csv(filepath)
      x = dataset.drop(['ID'], axis=1).values
      ID = np.array(dataset['ID'].values)
      # y_test = np.array(dataset['y'][split_index:][1:].values.reshape(-1, 64))
      return x, ID
    raise NotImplementedError()


############################################
#####        Helper functions          #####
############################################

def plot_dataset(X, y):
    '''
    This function generates the plot to visualize the dataset  

    Args:
      X : 2D numpy array of data points. Dimensions (n x 64)
      y : 2D numpy array of target values. Dimensions (n x 1)

    Returns:
      None
    '''
    plt.title('Plot of the unknown dataset')
    cmap = plt.cm.get_cmap('tab20', 64)
    
    for i in range(0, 64):
        plt.scatter(X[:,0], y, color=cmap(i), label=f'Feature {i}')
  
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.show()

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Starting experiment #####")
    
    print(RESET +  "Loading dataset: ",end="")
    try:
        X_train, y_train = read_dataset('train.csv')
        X_test, ID = read_dataset('test.csv')
        # X_train, y_train, X_validation, y_validation = read_dataset('train.csv')
        # X_test, ID = read_dataset('test.csv')

        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()

    print(RESET +  "Plotting dataset: ",end="")
    try:
        # plot_dataset(X_train, y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Performing input transformation: ", end="")
    try:
        # X_train = transform_input(X_train)
        # X_test = transform_input(X_validation)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
        
    print(RESET + "Caclulating weights: ", end="")
    try:
        linear_reg = LinearRegression()
        linear_reg.fit(X_train,y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()

    print(RESET +  "Regularizing: ",end="")
    try:
        linear_reg.regularize(X_validation, y_validation)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Checking closeness: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        print(GREEN + "done")
        
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()

    print(RESET + "Exporting predicted values: ", end="")
    try:
        scores = np.round(y_hat).astype(int)
        f_scores = pd.DataFrame({'ID': ID, 'score': scores})
        f_scores.to_csv('scores.csv', index=False)
        print(GREEN + "done")
        
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()