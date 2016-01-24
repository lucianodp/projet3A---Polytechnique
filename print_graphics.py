import numpy as np
import matplotlib.pyplot as plt

#####-------------------------------------------
#####-------------------------------------------
#####-------------------------------------------

# Printing the data points (BLUE), the classified ones (RED for 0 and GREEN for 1) and the classification boundary
# Currently works in two dimentions only


def print_graphs(data, p, tr, clf):
    #print(clf.coef_)
    w = clf.coef_[0]      #vector orthogonal to the separating hyperplane
    a = -w[0] / w[1]

    x   ,    y = data[:,0], data[:,1]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    xx = np.linspace(xmin-0.5, xmax+0.5)
    yy = a * xx - clf.intercept_[0] / w[1]   ## w[0]*xx + w[1]*yy + intercept = 0

    p1, p2 = p[tr > 0], p[tr <= 0]
    
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(p1[:,0], p1[:,1], c='r')
    plt.scatter(p2[:,0], p2[:,1], c='g')
    plt.plot(xx, yy, 'k-')

    plt.xlim( (xmin-0.5, xmax+0.5) )
    plt.ylim( (ymin-0.5, ymax+0.5) )

    plt.show()

def print_graphs2(data, p, tr, clf):
    h = 0.02
    
    x   ,    y = data[:,0], data[:,1]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    p1, p2 = p[tr > 0], p[tr <= 0]
    
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(p1[:,0], p1[:,1], c='r')
    plt.scatter(p2[:,0], p2[:,1], c='g')

    plt.show()
