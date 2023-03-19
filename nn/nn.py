# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self ._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # output should match A_prev

        # calculate the layer linear transformed matrix, the dot product of the previous layer matrix and weights + bias
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T

        # use the calculated matrix to get the activation matrix for our next layer
        if activation == "relu":
            A_curr = self._relu(Z_curr)
        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # initialize cache with the input matrix (will be changed later, just need a matrix in there)
        cache = {'A0': X}
        # initialize first activation matrix
        A_prev = X

        for (index, layer) in enumerate(self.arch):
            # for each layer, get the weights, bias, and activation function
            # index starts from 0 but dictionary references beginning from 1, so add 1 to index
            W_curr = self._param_dict[f"W{index + 1}"]
            b_curr = self._param_dict[f"b{index + 1}"]
            activation = layer['activation']

            # forward for that layer
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            # store new A and Z to cache dictionary
            cache[f"A{index + 1}"] = A_curr
            cache[f"Z{index + 1}"] = Z_curr

            # initialize current A as previous A for the next layer iteration
            A_prev = A_curr

        # After all the layers
        return A_prev, cache



    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # results depend on activation function
        if activation_curr == 'sigmoid':
            backprop_activation = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'relu':
            backprop_activation = self._relu_backprop(dA_curr, Z_curr)

        # calculate derivatives
        dA_prev = np.dot(backprop_activation, W_curr)
        dW_curr = np.dot(backprop_activation.T, A_prev)
        db_curr = np.sum(backprop_activation, axis=0).reshape(b_curr.shape)

        # return output (derivatives)
        return dA_prev, dW_curr, db_curr



    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # start with prediction. how did it compare to actual output?
        # how does each layer (weight, bias) affect the deviation to the actual input?
        # then, change the layers to minimize deviation

        # initialize dictionary containing gradient information
        grad_dict = {}

        # calculate dA_curr based on the loss function
        if self._loss_func == 'bce':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mse':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)

        for (index, layer) in reversed(list(enumerate(self.arch))):
            # for each layer, get the weights, bias, and activation function
            # index starts from 0 but dictionary references beginning from 1, so add 1 to index
            W_curr = self._param_dict[f"W{index + 1}"]
            b_curr = self._param_dict[f"b{index + 1}"]
            activation = layer['activation']

            # get Z (linear transform) and previous A (activation) from the cache dictionary
            A_prev = cache[f"A{index}"]
            Z_curr = cache[f"Z{index + 1}"]
            # get activation function
            activation_function = layer["activation"]
            # run backpropagation
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_function)

            # store bias and weight results into dictionary
            grad_dict[f"db{index + 1}"] = db_curr
            grad_dict[f"dW{index + 1}"] = dW_curr
            # update activation as previous to repeat for following previous layer
            dA_curr = dA_prev


        # After all the layers, return weights/biases
        return grad_dict


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer_id in range(1, (len(self.arch) + 1)):
            # update self._param_dict, which is an internal attribute containing the parameters
            # update bias values, remember layer keys start from 1 so start from 1 in the iteration to reference keys properly
            # subtract the bias gradient in backprop from the bias in forward, multiplied by learning rate
            self._param_dict['b' + str(layer_id)] = self._param_dict["b" + str(layer_id)] - self._lr * grad_dict['db' + str(layer_id)]
            # same thing for weight!
            self._param_dict['W' + str(layer_id)] = self._param_dict["W" + str(layer_id)] - self._lr * grad_dict['dW' + str(layer_id)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # initialize empty lists for epoch losses (training and validation)
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # separate training data into batches
        # get number of batches based on the size of our data
        num_batches = np.ceil(len(y_train) / self._batch_size)

        # begin iterating for each epoch
        for i in range(self._epochs):

            # shuffle data
            shuffle_idx = np.random.permutation(len(y_train))
            shuffled_X_train = X_train[shuffle_idx]
            shuffled_y_train = y_train[shuffle_idx]

            # split shuffled training data into batches
            X_batch_train = np.array_split(shuffled_X_train, num_batches)
            y_batch_train = np.array_split(shuffled_y_train, num_batches)

            # keep track of loss for each entry
            epoch_loss = []

            # iterate
            for X, y in zip(X_batch_train, y_batch_train):

                # get forward results
                out, cache = self.forward(X)
                # calculate loss based on loss function
                if self._loss_func == 'bce':
                    # convert to np.float64 or else the log will not work
                    out_float64 = out.astype(float)
                    loss = self._binary_cross_entropy(y, out_float64)
                elif self._loss_func == 'mse':
                    loss = self._mean_squared_error(y, out)

                epoch_loss.append(loss)

                # backprop and update parameters based on backprop
                grad_dict = self.backprop(y, out, cache)
                self._update_params(grad_dict)

            # predict values from validation based on neural network
            pred = self.predict(X_val)
            # get epoch loss for validation
            if self._loss_func == 'bce':
                # convert to np.float64 or else the log will not work
                pred_float64 = pred.astype(float)
                val_loss = self._binary_cross_entropy(y_val, pred_float64)
            elif self._loss_func == 'mse':
                val_loss = self._mean_squared_error(y_val, pred)

            # append average epoch loss from training
            per_epoch_loss_train.append(np.mean(epoch_loss))
            # append average validation loss
            per_epoch_loss_val.append(np.mean(val_loss))

        return per_epoch_loss_train, per_epoch_loss_val


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # y_hat from feed forward, don't need the cache (second argument)
        return self.forward(X)[0]

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.e ** (-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # partial derivative of current layer
        return dA * (1 - self._sigmoid(Z)) * self._sigmoid(Z)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        Z = Z * (Z > 0)
        # for each element in Z, return the element if it is bigger than 0, otherwise return 0
        return Z

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        return (Z > 0) * dA

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return (((1 - y) / (1 - y_hat)) - (y / y_hat)) / len(y)

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean((y - y_hat) ** 2)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return (-2 * (y - y_hat)) / len(y)