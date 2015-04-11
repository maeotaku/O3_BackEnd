import numpy as np
import cv2
from matplotlib import pyplot as plt

def compareHistograms(h1, h2):
    return cv2.compareHist(h1, h2, 1)

def getHist(s, xmax):
    xmin = 0
    #xmax = 256#65536
    step = 1
    values, bin_edges = np.histogram(s, np.linspace(2, xmax, (xmax-xmin)/step))
    values = np.float32(values)
    #values.convertTo(values, cv2.CV_32F);
    return values, bin_edges
    '''
    nbins = values.size
    plt.bar(bin_edges[:-1], values, width=bin_edges[1]-bin_edges[0], color='red', alpha=0.5)
    #plt.xlim(1,256)
    plt.ylim(0,8000)
    plt.hist(s, bins=nbins, alpha=0.5)
    '''
def showHist(s, values, bin_edges):
    nbins = values.size
    plt.bar(bin_edges[:-1], values, width=bin_edges[1]-bin_edges[0], color='red', alpha=0.5)
    #plt.xlim(1,256)
    plt.ylim(0,8000)
    plt.hist(s, bins=nbins, alpha=0.5)
    plt.show()
    
'''        
def show(title):
    plt.suptitle(title)
    plt.show()
'''