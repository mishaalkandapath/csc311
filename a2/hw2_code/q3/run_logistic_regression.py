from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 100
    }
    weights = np.ones((train_inputs.shape[1] + 1, 1)) * 0.2
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    acc_vals_Tr = [] #storing cross entropy loss for the training set
    acc_vals_cv = [] #similarly for the validation set
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters) #getting the derivatives and current loss for training set
        fcv,dfcv,ycv = logistic(weights, valid_inputs, valid_targets, hyperparameters) #same for validation set
        weights = weights - hyperparameters["learning_rate"] * df #gradient descent, but only on the gradient on the loss in training
        #storing CE loss averaged for plotting purposes
        acc_vals_Tr += [f] 
        acc_vals_cv += [fcv]


    valid_y = logistic_predict(weights, valid_inputs) #final prediction on the validation set

    #printing all the prediction accuracies on training, test, and validation for the set of hyperparameters
    print(evaluate(train_targets, logistic_predict(weights, train_inputs))[1])    
    print(evaluate(valid_targets, valid_y)[1])
    print(evaluate(test_targets, logistic_predict(weights, test_inputs))[1])

    #plotting the change in CE over number of iterations for training and validation sets
    plt.plot([t for t in range(hyperparameters["num_iterations"])], acc_vals_Tr, label="training set CE")
    plt.plot([t for t in range(hyperparameters["num_iterations"])], acc_vals_cv, label="validation set CE")
    plt.xlabel("iterations")
    plt.ylabel("Averaged Cross Entropy")
    plt.legend()
    plt.savefig("large_dat.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
