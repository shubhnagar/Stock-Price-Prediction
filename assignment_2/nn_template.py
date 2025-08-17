import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Returns the ReLU value of the input x
def relu(x):
    return max(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x>0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

## TODO 1c: Return the derivative of the tanh value of the input x
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

## TODO 1d: Return the derivative of the tanh value of the input x
def tanh_derivative(x):
    return 1 - np.power(tanh(x), 2)

# Mapping from string to function
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Given a list of activation functions, the following function returns
# the corresponding list of activation functions and their derivatives
def get_activation_functions(activations):  
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

class NN:
    def __init__(self, input_dim, hidden_dims, activations=None):
        '''
        Parameters
        ----------
        input_dim : int
            size of the input layer.
        hidden_dims : LIST<int>
            List of positive integers where each integer corresponds to the number of neurons 
            in the hidden layers. The list excludes the number of neurons in the output layer.
            For this problem, we fix the output layer to have just 1 neuron.
        activations : LIST<string>, optional
            List of strings where each string corresponds to the activation function to be used 
            for all hidden layers. The list excludes the activation function for the output layer.
            For this problem, we fix the output layer to have the sigmoid activation function.
        ----------
        Returns : None
        ----------
        '''
        assert(len(hidden_dims) > 0)
        assert(activations == None or len(hidden_dims) == len(activations))
         
        # If activations is None, we use sigmoid activation for all layers
        if activations == None:
            self.activations = [sigmoid]*(len(hidden_dims)+1)
            self.activation_derivatives = [sigmoid_derivative]*(len(hidden_dims)+1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        ## TODO 2: Initialize weights and biases for all hidden and output layers
        ## Initialization can be done with random normal values, you are free to use
        ## any other initialization technique.
        self.weights = []
        self.biases = []

        self.weights.append(np.random.normal(0, 1, (input_dim, hidden_dims[0])))
        self.biases.append(np.random.normal(0, 1, (hidden_dims[0], 1)))
        for i in range (1, len(hidden_dims)):
                self.weights.append(np.random.normal(0, 1, (hidden_dims[i-1], hidden_dims[i])))
                self.biases.append(np.random.normal(0, 1, (hidden_dims[i], 1)))
        
        self.weights.append(np.random.normal(0, 1, (hidden_dims[len(hidden_dims)-1], 1)))
        self.biases.append(np.random.normal(0, 1, (1, 1)))

    def forward(self, X):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        ----------
        Returns : output probabilities, numpy array of shape (N, 1) 
        ----------
        '''
        # Forward pass

        ## TODO 3a: Compute activations for all the nodes with the corresponding
        ## activation function of each layer applied to the hidden nodes
        

        ## TODO 3b: Calculate the output probabilities of shape (N, 1) where N is number of examples
        self.z = []
        self.a = [X] 

        for i in range(len(self.weights) - 1):  
            z_curr = np.dot(self.a[i], self.weights[i]) + self.biases[i].T
            a_curr = self.activations[i](z_curr)
            self.z.append(z_curr)
            self.a.append(a_curr)
        
        z_out = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1].T
        output_probs = self.activations[-1](z_out) 
        self.z.append(z_out)
        self.a.append(output_probs)

        return output_probs

    def backward(self, X, y):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        y : target labels, numpy array of shape (N, 1) where N is the number of examples
        ----------
        Returns : gradients of weights and biases
        ----------
        '''
        # Backpropagation

        ## TODO 4a: Compute gradients for the output layer after computing derivative of 
        ## sigmoid-based binary cross-entropy loss
        ## Hint: When computing the derivative of the cross-entropy loss, don't forget to 
        ## divide the gradients by N (number of examples)  
        self.grad_weights = [0] * len(self.weights)
        self.grad_biases = [0] * len(self.biases)
        N = len(y)
        y_np = np.array(y)
        y_reshaped = y_np.reshape(-1, 1)

        d_output = -(y_reshaped / self.a[-1] - (1 - y_reshaped) / (1 - self.a[-1])) / N
        self.grad_weights[-1] = np.dot(self.a[-2].T, d_output)
        self.grad_biases[-1] = np.sum(d_output, axis=0, keepdims=True).T
        
        
        ## TODO 4b: Next, compute gradients for all weights and biases for all layers
        ## Hint: Start from the output layer and move backwards to the first hidden layer
        
        for i in reversed(range(len(self.weights) - 1)):
            d_hidden = np.dot(d_output, self.weights[i + 1].T) * self.activation_derivatives[i](self.z[i])
            self.grad_weights[i] = np.dot(self.a[i].T, d_hidden)
            self.grad_biases[i] = np.sum(d_hidden, axis=0, keepdims=True).T
            d_output = d_hidden

        return self.grad_weights, self.grad_biases

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                gd_flag: 1 for Vanilla GD, 2 for GD with Exponential Decay, 3 for Momentum
                momentum: Momentum coefficient, used when gd_flag is 3.
                decay_constant: Decay constant for exponential learning rate decay, used when gd_flag is 2.
            epoch: Current epoch number
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        ### Calculate updated weights using methods as indicated by gd_flag

        ## TODO 5a: Variant 1(gd_flag = 1): Vanilla GD with Static Learning Rate
        ## Use the hyperparameter learning_rate as the static learning rate

        ## TODO 5b: Variant 2(gd_flag = 2): Vanilla GD with Exponential Learning Rate Decay
        ## Use the hyperparameter learning_rate as the initial learning rate
        ## Use the parameter epoch for t
        ## Use the hyperparameter decay_constant as the decay constant

        ## TODO 5c: Variant 3(gd_flag = 3): GD with Momentum
        ## Use the hyperparameters learning_rate and momentum

        if 'velocity_W' not in optimizer_params:
            optimizer_params['velocity_W'] = [np.zeros_like(w) for w in weights]
        if 'velocity_B' not in optimizer_params:
            optimizer_params['velocity_B'] = [np.zeros_like(b) for b in biases]

        updated_W = []
        updated_B = []

        if (gd_flag == 1):
            for i in range(len(weights)):
                updated_W.append(weights[i] - learning_rate * delta_weights[i])
                updated_B.append(biases[i] - learning_rate * delta_biases[i])
        elif (gd_flag == 2):
            for i in range(len(weights)):
                updated_W.append(weights[i] - learning_rate * delta_weights[i])
                updated_B.append(biases[i] - learning_rate * delta_biases[i])

            optimizer_params['learning_rate'] = learning_rate * np.exp(-1 * decay_constant * epoch)
        elif (gd_flag == 3):
            for i in range(len(weights)):
                optimizer_params['velocity_W'][i] = momentum * optimizer_params['velocity_W'][i] - learning_rate * delta_weights[i]
                updated_W.append(weights[i] + optimizer_params['velocity_W'][i])
            
                optimizer_params['velocity_B'][i] = momentum * optimizer_params['velocity_B'][i] - learning_rate * delta_biases[i]
                updated_B.append(biases[i] + optimizer_params['velocity_B'][i])
        
        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                beta: Exponential decay rate for the first moment estimates.
                gamma: Exponential decay rate for the second moment estimates.
                eps: A small constant for numerical stability.
        '''
        learning_rate = optimizer_params['learning_rate']
        beta1 = optimizer_params['beta1']
        beta2 = optimizer_params['beta2']
        eps = optimizer_params['eps']       

        ## TODO 6: Return updated weights and biases for the hidden layer based on the update rules for Adam Optimizer
        updated_W = []
        updated_B = []

        if 'm_W' not in optimizer_params:
            optimizer_params['m_W'] = [np.zeros_like(w) for w in weights]
        if 'm_B' not in optimizer_params:
            optimizer_params['m_B'] = [np.zeros_like(b) for b in biases]
        if 'v_W' not in optimizer_params:
            optimizer_params['v_W'] = [np.zeros_like(w) for w in weights]
        if 'v_B' not in optimizer_params:
            optimizer_params['v_B'] = [np.zeros_like(b) for b in biases]
    
        if 't' not in optimizer_params:
            optimizer_params['t'] = 1
        else:
            optimizer_params['t'] += 1
        t = optimizer_params['t']
    
        for i in range(len(weights)):
            optimizer_params['m_W'][i] = beta1 * optimizer_params['m_W'][i] + (1 - beta1) * delta_weights[i]
            optimizer_params['m_B'][i] = beta1 * optimizer_params['m_B'][i] + (1 - beta1) * delta_biases[i]
        
            optimizer_params['v_W'][i] = beta2 * optimizer_params['v_W'][i] + (1 - beta2) * (delta_weights[i] ** 2)
            optimizer_params['v_B'][i] = beta2 * optimizer_params['v_B'][i] + (1 - beta2) * (delta_biases[i] ** 2)

            m_hat_W = optimizer_params['m_W'][i] / (1 - beta1 ** t)
            m_hat_B = optimizer_params['m_B'][i] / (1 - beta1 ** t)
        
            v_hat_W = optimizer_params['v_W'][i] / (1 - beta2 ** t)
            v_hat_B = optimizer_params['v_B'][i] / (1 - beta2 ** t)
        
            updated_W.append(weights[i] - learning_rate * m_hat_W / (np.sqrt(v_hat_W) + eps))
            updated_B.append(biases[i] - learning_rate * m_hat_B / (np.sqrt(v_hat_B) + eps))

        return updated_W, updated_B

    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Divide X,y into batches
            X_batches = np.array_split(X_train, X_train.shape[0]//batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0]//batch_size)
            for X, y in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X)
                # Backpropagation and gradient descent weight updates
                dW, db = self.backward(X, y)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)

            # Compute the training accuracy and training loss
            train_preds = self.forward(X_train)
            train_loss = np.mean(-y_train*np.log(train_preds) - (1-y_train)*np.log(1-train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            train_losses.append(train_loss)

            # Compute the test accuracy and test loss
            test_preds = self.forward(X_eval)
            test_loss = np.mean(-y_eval*np.log(test_preds) - (1-y_eval)*np.log(1-test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)

        return train_losses, test_losses

    
    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses, optimizer, optimizer_params):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if optimizer == "bgd":
            plt.savefig(f'loss_bgd_' + str(optimizer_params['gd_flag']) + '.png')
        else:
            plt.savefig(f'loss_adam.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    # csv_file_path = "data_train.csv"
    # eval_file_path = "data_eval.csv"

    csv_file_path = "assgmt2/data_train.csv"
    eval_file_path = "assgmt2/data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X_train.shape[1]
    X_train = X_train**2
    X_eval = X_eval**2
    hidden_dims = [4, 5, 2] # the last layer has just 1 neuron for classification
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid', 'sigmoid']

    ## vanilla gd
    optimizer = "bgd"
    optimizer_params = {
        'learning_rate': 0.05,
        'gd_flag': 1,
        'momentum': 0.0003,
        'decay_constant': 0.05
    }

    ## Vanilla gd with decay
    # optimizer = "bgd"
    # optimizer_params = {
    #     'learning_rate': 0.1,
    #     'gd_flag': 2,
    #     'momentum': 0.0003,
    #     'decay_constant': 0.05
    # }

    # GD with momentum
    # optimizer = "bgd"
    # optimizer_params = {
    #     'learning_rate': 0.01,
    #     'gd_flag': 3,
    #     'momentum': 0.0003,
    #     'decay_constant': 0.05
    # }
    
    # For Adam optimizer you can use the following
    # optimizer = "adam"
    # optimizer_params = {
    #     'learning_rate': 0.005,
    #     'beta1' : 0.3,
    #     'beta2' : 0.8,
    #     'eps' : 1e-3
    # }

     
    model = NN(input_dim, hidden_dims)
    train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
    test_preds = model.forward(X_eval)

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)
