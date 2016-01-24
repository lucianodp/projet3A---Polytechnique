from print_graphics import *
import numpy as np
from sklearn.svm import SVC

# in this function, we sample the 'size'-closest points to the current classification boundary
# and ask the user to classify them.

def sampling_and_classify(data, size, clf, sampled_data):
    # the first step here is to order the points according to their distance to the
    # separating boundary.

    ordered_data = []
    
    #loopin through the data
    for i in range(len(data)):
	#current data point
        d = data[i].reshape(1,-1) 

 	#distance to separating boundary
        dist = abs( clf.decision_function(d)[0] ) 

	#storing the distance of each point - pair (index,distance)
        ordered_data.append( (i, dist) ) 

        
    #ordering dy decreasing order of distances
    ordered_data.sort(key = lambda pair: pair[1]) 


    #we'll classify point until we have 'size' points, or the data vector ends
    i, sample_id, classify = 0, [], []
    
    while(i < len(ordered_data) and len(classify) < size):

        #current point        
        pair = ordered_data[i]

	#if the current point has not already been sampled, we ask the user to classify it
        if(pair[0] not in sampled_data):

            print(data[pair[0]], pair[1])
            
	    #asking the question
            resp = int(input("Classification time! Quick, 0 (no), 1 (yes) or 2 (don't know)? "))
            
            
	    #if he does not know the answer, we just skip to the next iteration
            
            if(resp != 2): 
                sample_id.append(pair[0]) #classified point index
                classify.append(resp)     #classification
                sampled_data.add(pair[0]) #add index to classified points list
                
            print("")

        i+=1
            
    return (np.array(sample_id), np.array(classify))


#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------

# Sampling new points, classifying then and changing the borders

def iteration(data, sample, y, clf, sampled_data):
    
    (new_sample, new_y) = sampling_and_classify(data, 5, clf, sampled_data)
    
    sample = np.concatenate([sample, new_sample])
    y      = np.concatenate([y, new_y])
    
    p = data[sample]
        
    clf = SVC(kernel = "linear")
    clf.fit(p, y)
    
    print_graphs(data, p, y, clf)
    
    return (sample, y, clf)
    
