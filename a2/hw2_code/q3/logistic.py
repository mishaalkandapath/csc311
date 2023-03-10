from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    ones = np.ones((data.shape[0], 1)) 
    data = np.concatenate((data, ones), axis=1) #adding a column of ones in the last col of the data matrix
    y = np.matmul(data, weights) #matrix mutliplying the design and weight matrix for linear value
    y = sigmoid(y) #signmoided the value for output range in 0-1
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    entropy_term = (((-1) * targets) * (np.log2(y))) - ((1-targets)*(np.log2(1-y))) #formula explained in class
    ce = np.sum(entropy_term)/targets.shape[0] #average cross entropy
    classes = np.where(y >= 0.5, 1, 0) #everything with more than or equal to to chance probability is class 1, rest 0
    frac_correct = np.count_nonzero(np.where(classes == targets, 1, 0))/targets.shape[0] #counting and dividing the number of correctly classed samples (class == targets) with the total number of samples
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data) 

    #####################################################################
    # TODO:                       r                                      #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    ones = np.ones((data.shape[0], 1)) 
    data = np.append(data, ones, axis=1) #appending a col of ones
    f = evaluate(targets, y)[0] #getting the average cross entropy loss given outputs from predictor
    df = (np.matmul(data.T, y-targets)) / data.shape[0] #gradient formula: (X.T(y - targets)/N)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
