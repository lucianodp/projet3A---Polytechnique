from first_sampling import *
from sampling_and_classify import *
from print_graphics import *
from dimentionality_reduction import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def generate(mu1, sigma1, mu2, sigma2, size):
    x1 = np.random.normal(mu1, sigma1, size)
    x2 = np.random.normal(mu2, sigma2, size)
    y1 = np.random.normal(mu1, sigma1, size)
    y2= np.random.normal(mu2, sigma2, size)

    a = np.concatenate([x1,x2])
    b = np.concatenate([y1,y2])
    c = np.vstack((a,b))

    #random data between [-5,5] in another direction
    for i in range(1):
        z = 10*np.random.rand(2*size) - 5
        c = np.vstack((c, z))

    d = np.transpose(c)
    return d

    
#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------
    
def main():
    sampled_data = set([])
    
    data = generate(-3,1,1,1,1000)
    #print(len(data))
    
    pca = PCA()
    pca.fit(data)
    #print(data[list(sampled_data)])
    evr = pca.explained_variance_ratio_
    print("evr: ")
    print(evr)
    print(pca.components_)
    
    (sample, y, clf) = first_iteration(data, sampled_data)
    #print(sample)
    #sample_new = SelectKBest(chi2, k = 3).fit_transform(data[sample], y)
    #print(sample_new)
    #we can gather only this submatrix corresponding to data already classified
    #print(data[list(sampled_data)])
    #pca_dec(data, sampled_data)

    
    par = int(input("stop? (0 stop, 1 continues) "))
    print("")

    while(par == 1 and len(sample) > 0):
        (sample, y, clf) = iteration(data, sample, y, clf, sampled_data)
        #pca_dec(data, sampled_data)
        
        par = int(input("Stop? (0 stops, 1 continues) "))
        print("")


if __name__ == "__main__":
    main()
