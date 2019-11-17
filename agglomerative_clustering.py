import numpy as np
import pandas as pd
class AgglomerativeClustering:
    def __init__(self, df, n_cluster, linkage):

        # Init each data as a cluster
        df['Cluster'] = pd.DataFrame(np.arange(len(df)), columns=['Cluster'])
        
        # Init important variable
        self.arr = np.array(df)
        self.idxCluster = self.arr.shape[1] - 1
        self.cntCluster = self.arr.shape[0]
        self.n_cluster = n_cluster
        self.linkage = linkage
        self.m = np.zeros((self.cntCluster, self.cntCluster))
        self.initDistanceMatrix()

    def euclidean(self, row1, row2):
        return np.sqrt(np.sum((row1[:-1] - row2[:-1]) ** 2, axis = 0))
    
    def initDistanceMatrix(self):
        for i in range(self.m.shape[0]):
            for j in range(self.m.shape[0]):
                self.m[i, j] = self.euclidean(self.arr[i], self.arr[j])
        
    def printAll(self):
        print(self.arr)
        print(self.m)
    
    def findIdxCluster(self, idx):
        return np.where(self.arr[:, self.idxCluster] == idx)[0][0]
        
    def makeCluster(self, idx1, idx2):
        
        # Change cluster
        idxChange = np.where(self.arr[:, self.idxCluster] == idx2)
        
        for i in idxChange:
            self.arr[i, self.idxCluster] = self.arr[idx1, self.idxCluster]
        
        # Update cluster num
        idxCluster = np.where(self.arr[:, self.idxCluster] > idx2)
        
        for j in idxCluster:
            self.arr[j, self.idxCluster] -= 1
        
        # Update count cluster
        self.cntCluster -= 1
    
    def isMoreThanOne(self, num, arr):
        cnt = 0
        for i in arr:
            if i == num:
                cnt += 1
        return cnt > 1
    
    def checkLinkage(self):
        linkage = ['single', 'complete', 'average', 'average-group']
        return self.linkage in linkage
            
    def fit_predict(self):
        if self.checkLinkage():
            if self.n_cluster > 1:
                while (self.cntCluster > self.n_cluster):

                    # Create new cluster
                    minVal = np.min(self.m[self.m > 0])
                    idx = np.where(np.isclose(self.m, minVal))
                    self.makeCluster(idx[0][0], idx[1][0])

                    # Create new distance matrix
                    self.m = np.zeros((self.cntCluster, self.cntCluster))

                    # i, j are cluster label
                    for i in range(self.m.shape[0]):
                        for j in range(self.m.shape[0]):
                            if (i == j):
                                self.m[i, j] = 0
                            else:
                                # ONE TO ONE
                                if not self.isMoreThanOne(i, self.arr[:, self.idxCluster]) and not self.isMoreThanOne(j, self.arr[:, self.idxCluster]):
                                    self.m[i, j] = self.euclidean(self.arr[self.findIdxCluster(i)], self.arr[self.findIdxCluster(j)])
                                else:
                                    both = False
                                    # MANY TO MANY
                                    if self.isMoreThanOne(i, self.arr[:, self.idxCluster]) and self.isMoreThanOne(j, self.arr[:, self.idxCluster]):
                                        row, col = np.where(self.arr == i)
                                        idxI = row[np.where(col == self.idxCluster)]
                                        row, col = np.where(self.arr == j)
                                        idxJ = row[np.where(col == self.idxCluster)]
                                        temp = []
                                        for k in idxI:
                                            for l in idxJ:
                                                temp.append(self.euclidean(self.arr[k], self.arr[l]))
                                        both = True                                               
                                    # ONE TO MANY/MANY TO ONE
                                    elif self.isMoreThanOne(i, self.arr[:, self.idxCluster]):
                                        row, col = np.where(self.arr == i)
                                        idx = row[np.where(col == self.idxCluster)]
                                        temp = []
                                        for k in idx:
                                            temp.append(self.euclidean(self.arr[k], self.arr[self.findIdxCluster(j)]))
                                    elif self.isMoreThanOne(j, self.arr[:, self.idxCluster]):
                                        row, col = np.where(self.arr == j)
                                        idx = row[np.where(col == self.idxCluster)]
                                        temp = []
                                        for k in idx:
                                            temp.append(self.euclidean(self.arr[k], self.arr[self.findIdxCluster(i)]))
                                    if self.linkage == 'single':
                                        self.m[i, j] = np.min(temp)
                                    elif self.linkage == 'complete':
                                        self.m[i, j] = np.max(temp)
                                    elif self.linkage == 'average':
                                        self.m[i, j] = np.mean(temp)
                                    elif self.linkage == 'average-group':
                                        if both:
                                            for k in range(0, len(idxI)):
                                                for l in range(k + 1, len(idxI)):
                                                    temp.append(self.euclidean(self.arr[idxI[k]], self.arr[idxI[l]]))
                                            for k in range(0, len(idxJ)):
                                                for l in range(k + 1, len(idxJ)):
                                                    temp.append(self.euclidean(self.arr[idxJ[k]], self.arr[idxJ[l]]))
                                        else:
                                            for k in range(0, len(idx)):
                                                for l in range(k + 1, len(idx)):
                                                    temp.append(self.euclidean(self.arr[idx[k]], self.arr[idx[l]]))
                                        self.m[i, j] = np.mean(temp)
            else:
                self.arr[:, self.idxCluster] = 0
        else:
            print('Wrong Argument!')
        
#         self.printAll()
        
        return self.arr[:, self.idxCluster].astype('int32')
        
if __name__ == "__main__":
    np.random.seed(8)
    df = pd.DataFrame(np.random.rand(5, 4))
    print(AgglomerativeClustering(df, 2, 'average').fit_predict())
