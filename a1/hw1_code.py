import random
import numpy as np
import sklearn.feature_extraction.text as sk
import sklearn.model_selection as model_selec
import sklearn.tree as tree
from math import floor, log2
import matplotlib.pyplot as plt
from pprint import pprint


def cal_dataset_entropy(values):
    #assuming only 0 or 1 dataset, as is in this case
    samples = len(values)
    zeroes = values.tolist().count(0)
    ones = values.tolist().count(1)
    p_1 = ones/samples
    p_2 = zeroes/samples
    return  -p_1*log2(p_1) - p_2*log2(p_2)

def load_data(): 
    #load the fake and real headlines
    random.seed(10)
    fake_headlines = open("clean_fake.txt")
    real_headlines = open("clean_real.txt")

    #get a list of all the lines in each file
    fake_headlines = fake_headlines.readlines()
    real_headlines = real_headlines.readlines()
    #combines the list
    headlines = fake_headlines + real_headlines
    headlines = np.asarray(headlines) #convert to a numpy array
    #construct the labels: 0 for fake, 1 for real.
    values = [0]*len(fake_headlines) + [1]*len(real_headlines)
    values = np.asarray(values)
    
    vectorizer = sk.TfidfVectorizer(dtype=np.int64)#sk.CountVectorizer(dtype=np.int64) 
    X = vectorizer.fit_transform(headlines).toarray() # preprocess the text headlines

    trainX, testX, trainY, testY = model_selec.train_test_split(X, values, test_size=0.15, random_state=10, shuffle=True)
    trainX, cvX, trainY, cvY = model_selec.train_test_split(trainX, trainY, test_size=0.15/0.85, random_state=10, shuffle=True)

    dataset_entropy = cal_dataset_entropy(trainY) # cal the inherent entropy of the dataset
    print("Dataset entropy: ", dataset_entropy)
    for word in ["trump", "donald", "hillary", "and", "this", "the"]:
        print("Information gain for {}: {}".format(word ,compute_information_gain(dataset_entropy, word, trainX, trainY, vectorizer.get_feature_names_out()))) 

    return trainX, testX, cvX, trainY, testY, cvY, vectorizer.vocabulary_

def select_model():
    trainX, testX, cvX, trainY, testY, cvY, vocab = load_data()
    depths = [5, 7, 8, 9, 10]
    errors = []# list of errors for thr CV set
    models = [] #list of models for each depth
    for max_depth in depths:
        model = tree.DecisionTreeClassifier(max_depth = max_depth) # gini first
        model.fit(trainX, trainY)
        errors += [validate(cvX, cvY, model)]
        print("CV Accuracy for gini with depth {}: {}".format(max_depth, validate(cvX, cvY, model)))
        models.append(model)

    gini_depth = depths[errors.index(max(errors))] #get the index of the minimum error, and use that depth
    model_gini = models[errors.index(max(errors))] #gini using that max_depth
    gini_least_error = errors.index(max(errors))
    print("Test Accuracy {} for gini selected hyperparam {}".format(validate(testX, testY, model_gini), gini_depth))

    plt.plot(depths, errors, label="gini") #plot the depths vs errors plot for gini

    errors = [] # reinit errors
    models = []
    
    for max_depth in depths:
        model = tree.DecisionTreeClassifier(criterion="entropy",max_depth = max_depth) #information gain model
        model.fit(trainX, trainY)
        validate(cvX, cvY, model)
        errors += [validate(cvX, cvY, model)]
        models.append(model)
        print("CV Accuracy for entropy/information gain with depth {}: {}".format(max_depth, validate(cvX, cvY, model)))
    
    plt.plot(depths, errors, label="entropy")#plot the depth vs erorr for information gain
    
    entropy_depth = depths[errors.index(min(errors))] #get the depth with the least CV error
    model_entropy = models[errors.index(min(errors))]
    entropy_least_error = errors.index(min(errors))
    print("Test Accuracy {} for entropy/information gain selected hyperparam {}".format(validate(testX, testY, model_entropy), entropy_depth))


    errors = [] # reinit error
    models = []
    for max_depth in depths:
        model = tree.DecisionTreeClassifier(criterion="log_loss" ,max_depth = max_depth)  #log_loss model.
        model.fit(trainX, trainY)
        validate(cvX, cvY, model)
        errors += [validate(cvX, cvY,  model)]
        models.append(model)
        print("CV Accuracy for log_loss with depth {}: {}".format(max_depth, validate(cvX, cvY, model)))
    
    plt.plot(depths, errors, label="log_loss")
    
    logloss_depth = depths[errors.index(max(errors))] # get the depth with least CV error
    model_logloss = models[errors.index(max(errors))]
    logloss_least_error = errors.index(max(errors))
    print("Test Accuracy {} for logloss selected hyperparam {}".format(validate(testX, testY, model_logloss), logloss_depth))

    plt.legend()
    plt.savefig("models.png")

    model_errors = (gini_least_error, logloss_least_error, entropy_least_error) # accuracies
    #choose model to plot based on least test data error
    if max(model_errors) == gini_least_error:
        print("{} model chosen on CV accuracy".format("gini"))
        tree.plot_tree(model_gini, feature_names=sort_dict(vocab), max_depth=2,class_names=["Fake", "Real"])
        print("test accuracy {} for model {}".format(validate(testX, testY, model_gini), "gini"))
    elif max(model_errors) == logloss_least_error:
        print("{} model chosen on CV accuracy".format("logloss"))
        tree.plot_tree(model_logloss, feature_names=sort_dict(vocab), max_depth=2,class_names=["Fake", "Real"])
        print("test accuracy {} for model {}".format(validate(testX, testY, model_logloss), "logloss"))
    else:
        print("{} model chosen on CV accuracy".format("entropy"))
        tree.plot_tree(model_entropy, feature_names=sort_dict(vocab), max_depth=2,class_names=["Fake", "Real"])
        print("test accuracy {} for model {}".format(validate(testX, testY, model_entropy), "information gain"))
    
    plt.savefig("tree.png", dpi=600)



def validate(inputs, labels, model):
    #compute the error on trained model given inputs and labels
    values = model.predict(inputs).flatten() #predict the labels, and flattenm the array
    errors = np.abs(values - labels) # count the error by subtraction
    errors = errors[errors != 0] # get all the wrong label cases
    return  1 - (np.sum(errors) / len(labels)) #accurace = 1 - mislabel_proportion

def sort_dict(dictionary):
    #sort a dictionary of values based on their values (which are integer indexes in this case)
    values = []
    for key in dictionary:
        values.append((dictionary[key], key))
    values = sorted(values)
    for idx in range(len(values)):
        values[idx] = values[idx][1]
    return values #list of words sorted by index

def compute_information_gain(dataset_entropy, feature, headlines, values, features):
    #to find I(Y, feature) = H(Y) - H(Y|feature)
    if feature not in features:
        return 0 # no new information is learnt, H(Y) = H(Y|X)

    feature_idx = features.tolist().index(feature)
    mask = (headlines[:, feature_idx] > 0) #get all the inputs with the feature present
    trump_headlines = headlines[mask, :]
    nums_in = np.count_nonzero(mask)
    nums_out = headlines.shape[0] - nums_in
    sample_size = headlines.shape[0]

    values = np.asarray(values)
    values_in = values[mask] # all the labels of entries in split
    nums_in_real = np.count_nonzero(values_in) #real ones
    nums_in_fake = np.count_nonzero(values_in == 0) #fake ones

    nums_out_real = np.count_nonzero(values[(headlines[:, feature_idx] == 0)]) # real ones outside split
    nums_out_fake = values[(headlines[:, feature_idx] == 0)].shape[0] - nums_out_real #fake ones outside split
    H_y0 = -(nums_in_fake/(nums_in_fake + nums_in_real))*log2(nums_in_fake/(nums_in_fake + nums_in_real)) #entropy for Y given X s.t feature is in and value is fake
    H_y1 = -(nums_in_real/(nums_in_real + nums_in_fake))*log2(nums_in_real/(nums_in_real + nums_in_fake)) #entropy for Y given X s.t feature is in and value is real
    h_y0 = -(nums_out_fake/(nums_out_fake + nums_out_real))*log2(nums_out_fake/(nums_out_real + nums_out_fake)) #entropy for Y given X s.t feature is out and value is fake
    h_y1 = -(nums_out_real/(nums_out_fake + nums_out_real))*log2(nums_out_real/(nums_out_fake + nums_out_real)) #entropy for Y given X s.t feature is out and value is real

    return dataset_entropy - (nums_in/(sample_size))*(H_y0 + H_y1) - (nums_out/(sample_size))*(h_y0 + h_y1)

if __name__ == "__main__":
    select_model()









