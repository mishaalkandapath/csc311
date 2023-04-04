'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means: for each class, what is the mean value of each of the 64 features in the dataset
    for i in range(10):
        class_data = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(class_data, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances E[XY] - E[X]E[Y]
    for i in range(10):
        class_data = data.get_digits_by_label(train_data, train_labels, i)
        class_covar = np.zeros((64, 64))
        #subtract the feature-wise mean for each feature
        class_data = class_data - compute_mean_mles(train_data, train_labels)[i,:]
        #compute the covariance matrix
        class_covar = class_data.T @ class_data
        covariances[i] = class_covar/(class_data.shape[0] - 1)

    #adding the numerica stability factor:
    covariances = covariances + 0.01 * np.identity(64)
    return covariances

def compute_data_over_class(digits, means, covariance, label):
    ret_matrix = np.zeros((1, digits.shape[0]))
    #set all non-diag elements of covariance to 0 
    # covariance[label] = np.diag(np.diag(covariance[label])) #for part 4c
    det = np.linalg.det(covariance[label])
    const_factor = -32 * np.log((2 * np.pi)) -0.5* np.log(det)
    i = 0
    for xin in digits:
        xin = xin.reshape(-1, 1)
        # print(xin.shape, means[label].reshape(-1, 1).shape)
        main_factor = -0.5 * (xin - means[label].reshape(-1, 1)).T @ np.linalg.inv(covariance[label]) @ (xin - means[label].reshape(-1, 1))
        ret_matrix[0][i] = const_factor + main_factor
        i += 1

    return ret_matrix

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''

    return None

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    cond_like = np.zeros((digits.shape[0], 10))
    
    for i in range(10):
        cond_like[:, i] = compute_data_over_class(digits, means, covariances, i) * 0.01

    return cond_like

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    # Compute as described above and return
    return np.average(digits, axis=1)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    # Compute and return the most likely class
    print(np.argmax(digits, axis=1))
    return np.argmax(digits, axis=1)

def accuracy(predictions, labels):
    """ Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""
    # print(predictions.shape, labels.shape)
    acc = np.sum(predictions == labels) / labels.shape[0]
    return acc

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('a3/data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    log_like_train = conditional_likelihood(train_data, means, covariances)
    class_trains = classify_data(log_like_train, means, covariances)
    accuracy_train = accuracy(class_trains, train_labels)
    log_like_test  = conditional_likelihood(test_data, means, covariances)
    class_test = classify_data(log_like_test, means, covariances)
    accuracy_test = accuracy(class_test, test_labels)

    print("Average log likelihood on train {}".format(avg_conditional_likelihood(log_like_train, train_labels, means, covariances)))
    print("Average log likelihood on test {}".format(avg_conditional_likelihood(log_like_test, test_labels, means, covariances)))
    print("Accuracy on train {}".format(accuracy_train))
    print("Accuracy on test {}".format(accuracy_test))

if __name__ == '__main__':
    main()
