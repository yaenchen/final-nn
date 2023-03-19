# TODO: import dependencies and write unit tests below
from nn import (io, nn, preprocess)
import numpy as np
import pytest

# set seed
np.random.seed(10)
# instantiate simple neural network
net = nn.NeuralNetwork([{'input_dim': 5, 'output_dim': 10, 'activation': "sigmoid"},
                        {'input_dim': 10, 'output_dim': 5, 'activation': "sigmoid"}],
                        lr=0.1,
                        seed=10,
                        batch_size=5,
                        epochs=5,
                        loss_function='bce')
# make some data
X = np.random.randn(10, 5)

def test_single_forward():
    # establish weights and biases
    # run single forward
    A, Z = net._single_forward(W_curr=np.array([1, 2, 3]),
                               b_curr=np.array([1, 2, 3]),
                               A_prev=np.array([1, 0, 1]),
                               activation=net.arch[0]['activation'])
    # assert statements on example
    assert np.allclose(A, np.array([0.99330715, 0.99752738, 0.99908895]))
    assert np.allclose(Z, np.array([5, 6, 7]))

def test_forward():
    # run forward
    output, cache = net.forward(X)
    output = np.asarray(output, dtype='float64')
    expected = np.array([[0.48143897, 0.50816474, 0.53673459, 0.48668311, 0.46553634],
              [0.4838546, 0.51255378, 0.52826506, 0.4860597, 0.46110437],
              [0.4845847, 0.51192266, 0.53278144, 0.49032614, 0.46194268],
              [0.48202594, 0.50639076, 0.53848958, 0.48758986, 0.46632257],
              [0.48642051, 0.51055976, 0.53322558, 0.49056067, 0.46009727],
              [0.48592537, 0.50560177, 0.53700116, 0.48888211, 0.46386499],
              [0.48519459, 0.51070064, 0.53378183, 0.49235269, 0.46504538],
              [0.48324058, 0.51317822, 0.53094084, 0.48851457, 0.46299257],
              [0.47723994, 0.51165853, 0.53448979, 0.48483289, 0.46764646],
              [0.47845901, 0.51377405, 0.53337973, 0.48781133, 0.46761879]], dtype='float64')
    assert np.allclose(output, expected, "output doesn't match expected")

    assert np.allclose(cache['A1'], np.array([[0.44941055, 0.51806323, 0.46394821, 0.57990949, 0.52915659,
         0.46772466, 0.51249705, 0.55364032, 0.5211759 , 0.50744671],
        [0.43032647, 0.50636081, 0.4820492 , 0.6055673 , 0.7692477 ,
         0.52552173, 0.60044923, 0.54847189, 0.41318853, 0.5444526 ],
        [0.53269886, 0.50549652, 0.56304884, 0.53645759, 0.57739638,
         0.52691026, 0.54606135, 0.57274568, 0.45643006, 0.50268274],
        [0.47713266, 0.50751637, 0.46659241, 0.58886436, 0.56136322,
         0.44955951, 0.53091839, 0.54805681, 0.53907434, 0.50570233],
        [0.53679374, 0.51408618, 0.58224984, 0.51187655, 0.62980787,
         0.50888177, 0.56635873, 0.57042524, 0.42819278, 0.50371057],
        [0.47332013, 0.50155163, 0.43385536, 0.52490393, 0.57556408,
         0.49706507, 0.59712739, 0.48632787, 0.50378071, 0.4954399 ],
        [0.56521731, 0.46479458, 0.49461145, 0.54781893, 0.53921746,
         0.57089169, 0.59520677, 0.51251944, 0.51592061, 0.48936385],
        [0.46807423, 0.50551939, 0.48339095, 0.54078083, 0.50945733,
         0.5670774 , 0.54150277, 0.5379883 , 0.46917996, 0.5030566 ],
        [0.42166486, 0.51608772, 0.4599522 , 0.65084949, 0.52514767,
         0.47279625, 0.46239246, 0.59450681, 0.53982153, 0.52517852],
        [0.48530165, 0.49626289, 0.49078103, 0.61865684, 0.44164158,
         0.5322742 , 0.47110268, 0.5863504 , 0.54491215, 0.50636099]]), "A1 doesn't match expected")

def test_single_backprop():
    # make up parameters
    W_curr = np.array([1, 2, 3])
    b_curr = np.array([1, 2, 3])
    A_prev = np.array([1, 0, 1])
    Z_curr = np.array([5, 6, 7])
    dA_curr = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])
    # run single backprop
    dA_prev, dW_curr, db_curr = net._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, net.arch[0]['activation'])

    # assert all outputs
    assert np.allclose(dA_prev, np.array([0.01431174, 0.01431174, 0.01431174]))
    assert np.allclose(dW_curr, np.array([0.01329611, 0.00493302, 0.00182044]))
    assert np.allclose(db_curr, np.array([0.01994417, 0.00739953, 0.00273066]))

def test_predict():
    preds = net.predict(X)
    assert np.allclose(preds, np.array([[0.48143897, 0.50816474, 0.53673459, 0.48668311, 0.46553634],
       [0.4838546 , 0.51255378, 0.52826506, 0.4860597 , 0.46110437],
       [0.4845847 , 0.51192266, 0.53278144, 0.49032614, 0.46194268],
       [0.48202594, 0.50639076, 0.53848958, 0.48758986, 0.46632257],
       [0.48642051, 0.51055976, 0.53322558, 0.49056067, 0.46009727],
       [0.48592537, 0.50560177, 0.53700116, 0.48888211, 0.46386499],
       [0.48519459, 0.51070064, 0.53378183, 0.49235269, 0.46504538],
       [0.48324058, 0.51317822, 0.53094084, 0.48851457, 0.46299257],
       [0.47723994, 0.51165853, 0.53448979, 0.48483289, 0.46764646],
       [0.47845901, 0.51377405, 0.53337973, 0.48781133, 0.46761879]]))

def test_binary_cross_entropy():
    # simple test
    y_hat = np.array([0.5, 0.6, 0.2, 0.3, 0.6])
    y = np.array([1, 1, 1, 1, 1])

    bce = net._binary_cross_entropy(y, y_hat)
    assert bce == 0.9056418289703926, "BCE is not as expected!"

def test_binary_cross_entropy_backprop():
    # use previous y and y_hat
    y_hat = np.array([0.5, 0.6, 0.2, 0.3, 0.6])
    y = np.array([1, 1, 1, 1, 1])
    bce_backprop = net._binary_cross_entropy_backprop(y, y_hat)
    assert np.allclose(bce_backprop, np.array([-0.4, -0.33333333, -1., -0.66666667, -0.33333333])), "BCE backprop is not as expected!"

def test_mean_squared_error():
    y_hat = np.array([0.5, 0.6, 0.2, 0.3, 0.6])
    y = np.array([1, 1, 1, 1, 1])
    assert net._mean_squared_error(y,y_hat) == 0.34

def test_mean_squared_error_backprop():
    y_hat = np.array([0.5, 0.6, 0.2, 0.3, 0.6])
    y = np.array([1, 1, 1, 1, 1])
    assert np.allclose(net._mean_squared_error_backprop(y,y_hat), np.array([-0.2 , -0.16, -0.32, -0.28, -0.16])), "MSE backprop not as expected!"

def test_sample_seqs():
    """
    Unit test to make sure sequences are sampled properly (prevent class imbalance)
    """
    # run sample_seqs on some dummy data
    seqs, labels = preprocess.sample_seqs(['A', 'A', 'A', 'B'], [True,True,True,False])
    # check final list size is the average number of entries for each class
    assert len(seqs) == 4, "Final list size is not as expected"
    # make sure we get expected number
    assert seqs.count('A') == 2, "Expected number of sequences is wrong"
    assert labels.count(True) == 2, "Expected number of labels is wrong"


def test_one_hot_encode_seqs():
    """
    Unit test to make sure sequences are one-hot encoded
    """
    # initialize sequence
    seq_arr = ['AGA']
    # one-hot encode sequence
    encoded = preprocess.one_hot_encode_seqs(seq_arr)
    actual = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    assert (actual == encoded).all(), "Actual one-hot encoded sequence does not match expected."

    # initialize sequence
    seq_arr = ['TAGAA', "GAATT"]
    # one-hot encode sequence
    encoded = preprocess.one_hot_encode_seqs(seq_arr)
    assert len(encoded) == len(seq_arr)