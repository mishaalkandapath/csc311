from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T) 
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################

    neighbours = [1, 3, 5, 7, 9]
    accuracies = []
    acc1 = []
    for idx, neighbour in enumerate(neighbours):
        labels = knn(neighbour, train_inputs, train_targets, valid_inputs).flatten().tolist() #get the predicted class values from knn
        total_corr = sum(val == valid_targets[ix] for ix, val in enumerate(labels)) #compute the number of labels we got right
        accuracies += [total_corr/len(labels)] #divide by number of total datapoints to have predictec to get accuracy
        print(accuracies[idx])

        #using a similar procedure to find test accuracy, only doing this here rather than to do for only the chosen k value because the question asks to report
        # test accuracie for k-2 and k+2 too. Aware that test accuracy should be checked for only the final determined model with chosen hyperparameter k    
        test_labels = knn(neighbour, train_inputs, train_targets, test_inputs).flatten().tolist()
        total_corr = sum(val == test_targets[ix] for ix, val in enumerate(test_labels))
        acc1+= [total_corr/len(test_labels)]
        print("{} nearest neighbours: {}".format(neighbour, acc1[idx]))

    plt.plot(neighbours, accuracies, label="accuracies")
    plt.xlabel("neighbours")
    plt.ylabel("accuracies")
    plt.savefig("knn_acc.png")



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
