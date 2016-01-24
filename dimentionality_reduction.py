import numpy as np
from sklearn.decomposition import PCA

#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------

#try a first idea of pca
def pca_dec(data, sampled_data):
    pca = PCA()
    
    pca.fit(data[list(sampled_data)])
    #print(data[list(sampled_data)])
    evr = pca.explained_variance_ratio_
    print("evr: ")
    print(evr)
    print(pca.components_)

    t = pca.transform(data)
    #print(t)

    
    #get only the useful components
    components = []
    for i in range(len(pca.components_)):
        if evr[i] > 0.1:
            components.append(i)
            
    return t[:, components]
    
