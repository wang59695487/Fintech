import numpy as np
import math as m
from scipy.spatial.distance import cdist


#计算欧式距离
def distance(x1,x2):
    return np.sqrt(sum(np.power(x1-x2,2)))
#随机初始化类中心
def randcenter(set,k):
    dim=np.shape(set)[1]
    init_cen=np.zeros((k,dim))
    for i in range(dim):
        min_i=min(set[:,i])
        range_i=float(max(set[:,i]) - min_i)
        init_cen[:,i]=min_i + range_i*np.random.rand(k)
    return init_cen


#Kmeans主程序
def kmeans(X, k):
    '''
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    OUTPUT:  idx: cluster label, N-by-1 vector
    '''
    N, P = X.shape
    idx = np.zeros((N, 1))
    #划为k个子集并生成中心点
    center = randcenter(X, k)
    change = True
    while change:
        change = False
        for i in range(N):
            mindist = np.inf
            min_index = -1
            for j in range(k):
                distance1 = distance(center[j, :], X[i, :])
                if (distance1 < mindist):
                    mindist = distance1
                    min_index = j
            #中心点变化继续划分
            if idx[i] != min_index:
                change = True
            idx[i] = min_index
        for cen in range(k):
            cluster_data = X[np.nonzero(idx[:, 0] == cen)]
            center[cen, :] = np.mean(cluster_data, 0)

    #类型转换
    idx.astype(np.int16)

    pass
    return idx
#--------------------------------------------k-means++--------------------
#对一个样本找到与该样本距离最近的聚类中心
def nearest(point, cluster_centers):
    min_dist = np.inf
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist
#选择尽可能相距较远的类中心
def get_centroids(dataset, k):
    m, n = np.shape(dataset)
    cluster_centers = np.zeros((k , n))
    index = np.random.randint(0, m)
    cluster_centers[0,] = dataset[index, ]
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(dataset[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= np.random.rand()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all=sum_all - di
            if sum_all > 0:
                continue
            cluster_centers[i,] = dataset[j, ]
            break
    return cluster_centers
#主程序
def kmeans2(dataset,k):
    row_m=np.shape(dataset)[0]
    idx = np.zeros((row_m,1))
    center=get_centroids(dataset,k)
    change=True
    while change:
        change=False
        for i in range(row_m):
            mindist= np.inf
            min_index=-1
            for j in range(k):
                distance1=distance(center[j,:],dataset[i,:])
                if distance1<mindist:
                    mindist=distance1
                    min_index=j
            if idx[i,0] != min_index:
                change=True
            idx[i,:]=min_index
        for cen in range(k):
            cluster_data=dataset[np.nonzero(idx[:,0]==cen)]
            center[cen,:]=np.mean(cluster_data,0)
    return idx


#-------------------------------------------谱聚类--------------------

def spectral(W, k):
    '''
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    '''
    N = W.shape[0]
    idx = np.zeros(N, dtype=np.int)
    D = np.zeros(W.shape)
    for i in range(N):
        D[i][i] = np.sum(W[i][0:N])
    D_inv = np.linalg.inv(D)
    T = np.matmul(D_inv, D - W)
    eigval, eigvec = np.linalg.eig(T)
    eigval_real = np.zeros(eigval.shape)
    eigvec_real = np.zeros(eigvec.shape)
    for i in range(N):
        eigval_real[i] = float(eigval[i].real)
        for j in range(N):
             eigvec_real[i][j] = float(eigvec[i][j].real)
    ix = np.argsort(eigval_real)[0:k]
    X = eigvec_real[:, ix]
    idx = kmeans(X, k)
    return idx

def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')
    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]  # aj[i,i] = 0
        W[i, index] = 1
    W[aj >= threshold] = 0
    return W

