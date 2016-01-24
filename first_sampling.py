from print_graphics import *
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.mixture import GMM


# This function finds the first data sample
# We note that it returns only the INDEXES at the database, not the data points
# For now, we have only made a simple random sample throught the database

def first_sampling(data, size):
    sample = random.sample(range(0,len(data)),2*size)
    classifier = GMM(n_components = 4, covariance_type = 'full', init_params='wc', n_iter=20)
    classifier.fit(data)
    #print("Weights")
    #print(classifier.weights_)
    #print("Means ")
    #print(classifier.means_)

    proba = classifier.predict_proba(data)
    #print(proba)

    sample_id = []
    for i in sample:
        if((proba[i][0] < 0.00001 or proba[i][1] < 0.00001 or proba[i][2] < 0.00001 or proba[i][3] < 0.00001) and len(sample_id) < 10):
            sample_id.append(i)

    
    #O problema e que queriamos o indice, e nao o objeto
    return sample_id
    
#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------

def first_sampling_and_classify(data):
    sample = first_sampling(data, 10)

    cl = np.zeros( len(sample) )
    
    for i in range( len(sample) ):
        print( data[ sample[i] ] )
        
        r = int(input("Classification time! Quick, 0 ou 1? "))
        cl[i] = r
        
        print("")

    return (sample, cl)


#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------

def first_sampling_and_autoclassify(data):
    sample = first_sampling(data, 10)

    cl = np.zeros( len(sample) )
    
    for i in range( len(sample) ):
        
        if(data[sample[i]][2] < 700000):
            cl[i] = 1
        else:
            cl[i] = 0
        

    return (sample, cl)

#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------
    
    
def first_iteration(data, sampled_data):
    (sample, y) = first_sampling_and_autoclassify(data)

    sampled_data |= set(sample)
    
    p = data[sample]

    clf = SVC(kernel = "rbf")
    clf.fit(p, y)

    #print_graphs(data, p, y, clf)

    return (sample, y, clf)
    
    
