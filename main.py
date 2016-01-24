from first_sampling import *
from sampling_and_classify import *
from print_graphics import *
from dimentionality_reduction import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


def generate(mu1, sigma1, mu2, sigma2, size):
    x1 = np.random.normal(mu1, sigma1, size)
    x2 = np.random.normal(mu2, sigma2, size)
    y1 = np.random.normal(mu1, sigma1, size)
    y2= np.random.normal(mu2, sigma2, size)

    a = np.concatenate([x1,x2])
    b = np.concatenate([y1,y2])
    c = np.vstack((a,b))

    #random data between [-5,5] in another direction
    for i in range(18):
        z = 10*np.random.rand(2*size) - 5
        c = np.vstack((c, z))

    d = np.transpose(c)
    return d

    
def fselection(data, sample, y, k):
    anova_filter = SelectKBest(f_classif, k)
    anova_filter.fit(data[sample], y)
    return anova_filter.get_support()

    #toReturn = SelectKBest(f_classif, k).fit_transform(data[sample], y)
    #return toReturn

#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------
    
    
def main():
    sampled_data = set([])
    
    data = np.loadtxt("./data/real_estate.txt", dtype = None, delimiter = '\t', usecols = [2,3,4,5,6,7,8,9,10,11,12]) #generate(-4,1,2,1,1000)
    
    
    (sample, y, clf) = first_iteration(data, sampled_data)
    print (fselection(data, sample, y, 2))
    #print(sample)
    #sample_new = SelectKBest(chi2, k = 3).fit_transform(data[sample], y)
    #print(sample_new)
    #we can gather only this submatrix corresponding to data already classified
    #print(data[list(sampled_data)])
    #pca_dec(data, sampled_data)

    
    par = int(input("stop? (0 stop, 1 continue) "))
    print("")
    
    while(par == 1 and len(sample) > 0):
        
        (sample, y, clf) = iteration(data, sample, y, clf, sampled_data)
        #pca_dec(data, sampled_data)
        print (fselection(data, sample, y, 4))
        par = int(input("Stop? (0 stop, 1 continue) "))
        print("")

if __name__ == "__main__":
    main()
