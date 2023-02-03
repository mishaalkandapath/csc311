import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics.pairwise import manhattan_distances
import math

l2_norms = []
l1_dist = []
l2_norms_std = []
l1_dist_std = []

dims = [2** i for i in range(11)]

for dim in dims:
    #generate the points
    points = []
    for i in range(100):
        point = []
        for j in range(dim):
            point.append(random.random())
        points.append(point)
    
    eu_distances = [] #storing the euclidean distances across all pairs of points
    l1_distances = [] #storing l1 distances across all pairs of points
    for i in range(100): # across all datapoints
        for j in range(i+1, 100): #pair with another, except itself or previously paired with
            sum1 = 0 #distance for euclidean
            sum2 = 0 #l1 distance
            for x in range(dim): #for each distances
                sum1 += math.pow((points[i][x] - points[j][x]), 2) #square and add
                sum2 += abs((points[i][x] - points[j][x])) # add the absolute value

            eu_distances += [sum1] #this is the squared euclidean distance
            l1_distances += [sum2] 
    l2_norms += [np.mean(eu_distances)]
    l2_norms_std += [np.std(eu_distances)]
    l1_dist += [np.mean(l1_distances)]
    l1_dist_std += [np.std(l1_distances)]


plt.plot([i for i in range(11)], l2_norms, label="l2-mean")
plt.plot([i for i in range(11)], l1_dist, label="l1-mean")
plt.legend()
plt.show()

plt.plot([i for i in range(11)], l2_norms_std, label="std-l2")
plt.plot([i for i in range(11)], l1_dist_std, label="std-l1")
plt.legend()
plt.show()


