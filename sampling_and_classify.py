from print_graphics import *
import numpy as np
from sklearn.svm import SVC

# in this function, we sample the 'size'-closest points to the current classification boundary
# and ask the user to classify them.

def sampling_and_classify(data, size, clf, sampled_data):
    ordered_data = []
    
    for i in range(len(data)):
        d = data[i].reshape(1,-1)         #current data point
        dist = abs( clf.decision_function(d)[0] ) #distance to separating plane

        ordered_data.append( (i, dist) ) #including the pair (index, distance)

    ordered_data.sort(key = lambda pair: pair[1]) #ordering by distances

    i = 0
    sample_id = []
    classify = []

    #we'll classify point until we have 'size' points, or the data vector ends
    while(i < len(ordered_data) and len(classify) < size):
        pair = ordered_data[i]

        if(pair[0] not in sampled_data):
            print(data[pair[0]], pair[1])

            resp = int(input("Classification time! Quick, 0 (no), 1 (yes) or 2 (don't know)? "))

            if(resp != 2): #we include the classifications
                sample_id.append(pair[0]) #classified point index
                classify.append(resp)     #classification
                sampled_data.add(pair[0]) #add index to classified points list

            print("")
        i+=1
    
    return (np.array(sample_id), np.array(classify))

#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------

def sampling_and_autoclassify(data, size, clf, sampled_data):
    ordered_data = []
    
    for i in range(len(data)):
        d = data[i].reshape(1,-1)         #current data point
        dist = abs( clf.decision_function(d)[0] ) #distance to separating plane

        ordered_data.append( (i, dist) ) #including the pair (index, distance)

    ordered_data.sort(key = lambda pair: pair[1]) #ordering by distances

    i = 0
    sample_id = []
    classify = []

    #we'll classify point until we have 'size' points, or the data vector ends
    while(i < len(ordered_data) and len(classify) < size):
        pair = ordered_data[i]

        if(pair[0] not in sampled_data):
            
            sample_id.append(pair[0]) #classified point index/ always classify this point with either 0 or 1

            if(data[pair[0]][2] < 700000): #we include the classifications
                classify.append(1)     #classification
            else:
                classify.append(0)
                
            sampled_data.add(pair[0]) #add index to classified points list

            #print("")
        i+=1
    
    return (np.array(sample_id), np.array(classify))


#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------

# Sampling new points, classifying then and changing the borders

def iteration(data, sample, y, clf, sampled_data):
    
    (new_sample, new_y) = sampling_and_autoclassify(data, 5, clf, sampled_data)
    
    sample = np.concatenate([sample, new_sample])
    y      = np.concatenate([y, new_y])
    
    p = data[sample]
        
    clf = SVC(kernel = "rbf")
    clf.fit(p, y)
    
    #print_graphs(data, p, y, clf)
    
    return (sample, y, clf)
 
