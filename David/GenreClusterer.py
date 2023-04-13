## Made by David Gaviria    April 13, 2023 ##

from sklearn import model_selection
from sklearn import cluster


## Grouping class for clusters used by GenreClusterer
class Grouping:

    # init
    def __init__(self, _id):
        self._id = _id
        self.composition = dict()
        self.profile = []
        self.size = 0

    # add sample to profile
    def insert(self, sample, genre):
        if genre not in self.composition:
            self.composition[genre] = []  #insert corresponding point
        genreSlot = self.composition[genre]
        genreSlot.append(sample)
        self.size += 1

    # sort profile
    def computeProfile(self):
        for genre in self.composition:
            genrePercent = len(self.composition[genre]) / self.size
            self.profile.append((genre, genrePercent))
        self.profile.sort(key = lambda x:x[1], reverse=True)



## GenreClusterer class
class GenreClusterer:

    # init
    def __init__(self, k=8):
        self.model = None
        self.k = k
        self.clusters = {}
  
    # tune optimal parameters for model
    def find_parameters(self, xValidate, yValidate, crossVal=5, v=False):
        # find optimal parameter
        param_grid = {'n_clusters':[1,2,3,4,5,6,7,8,9,10]}
        grid = model_selection.GridSearchCV(cluster.KMeans(), param_grid, refit=True, verbose=v, cv=crossVal, n_jobs=-2) 
        grid.fit(xValidate, yValidate)
        # get best parameter and update model
        bestParameters = grid.best_params_
        self.k = bestParameters['n_clusters']
        return self.k
          
    # fit model
    def fit(self, xTrain, yTrain):
        self.model = cluster.KMeans(n_clusters=self.k)
        # create clusters 
        self.model.fit(xTrain)
        nLabels = self.model.labels_
        # build cluster profiles 
        for i in range(len(nLabels)):
            sampleIndex = xTrain.iloc[i]  #TODO: need to get sample index
            thisGenre = yTrain.iloc[i]
            clusterId = nLabels[i]
            # get cluster
            if clusterId not in self.clusters:
                self.clusters[clusterId] = Grouping(_id=i)
            thisCluster = self.clusters[clusterId] 
            # insert into cluster
            thisCluster.insert(sampleIndex, thisGenre)
        # compute the stats of the clusters
        for clusterId in self.clusters:
            thisCluster = self.clusters[clusterId]
            thisCluster.computeProfile()

    # print out clusters
    def displayClusters(self):
        for clusterId in self.clusters:
            thisCluster = self.clusters[clusterId]
            print("<Cluster %s>(%s)" % (thisCluster._id, thisCluster.profile))

            

