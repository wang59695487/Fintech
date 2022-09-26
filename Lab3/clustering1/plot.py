from numpy import *
import matplotlib.pyplot as plt


def plot(datas, cluster_assign, title):
    '''
    Show clustering results

    Input:  X: data point features, n-by-p maxtirx.
            idx: data point cluster labels, n-by-1 vector.
    '''
    plt.figure(figsize=(6, 6))
    plt.plot(datas[nonzero(cluster_assign == 0), 0], datas[nonzero(cluster_assign == 0), 1],'r.',markersize=5)
    plt.plot(datas[nonzero(cluster_assign == 1), 0], datas[nonzero(cluster_assign == 1), 1],'b.',markersize=5)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()