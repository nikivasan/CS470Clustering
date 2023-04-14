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
        otherPercentage = 0
        for genre in self.composition:
            genrePercent = len(self.composition[genre]) / self.size
            if genrePercent >= 0.05:
            # dont add genres that are insignificant
                self.profile.append((genre, genrePercent))
            else:
            #add it instead to 'other
                otherPercentage += genrePercent
        # add other percentage to make percentages = 1
        self.profile.append(('other', otherPercentage))     
        self.profile.sort(key = lambda x:x[1], reverse=True)



## GenreClusterer class
class GenreClusterer:

    # init
    def __init__(self, model):
        self.clusters = {}
        # get model type
        if model == 'KMeans':
            self.model = cluster.KMeans()
            self.modelType = 'KMeans'
        elif model == 'DBScan':
            self.model = cluster.DBSCAN()
            self.modelType = 'DBScan'
        else:
            raise Exception("Invalid model type, accepted models are 'KMeans' and 'DBScan'")

    # tune optimal parameters for model
    def find_parameters(self, xValidate, yValidate, crossVal=5, v=False):
        if self.modelType =='KMeans':
            # TODO: find max genre cap for k
            # find optimal parameter
            param_grid = {'n_clusters':[1,2,3,4,5,6,7,8,9,10]}
            grid = model_selection.GridSearchCV(cluster.KMeans(), param_grid, refit=True, verbose=v, cv=crossVal, n_jobs=-2) 
            grid.fit(xValidate, yValidate)
            # get best parameter and update model
            bestParameters = grid.best_params_
            self.model = cluster.KMeans(n_clusters=bestParameters['n_clusters'])
        else:
            # TODO: how to get optimal parameters, fix this
            bestParameters = self.model.get_params()
            self.model.set_params()
            # # TODO: generate eps list
            # # find optimal parameter
            # param_grid = {'eps':[0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10], 'min_samples':[1,2,3,4,5,6,7,8,9,10]}
            # grid = model_selection.GridSearchCV(cluster.DBSCAN(), param_grid, refit=True, verbose=v, cv=crossVal, n_jobs=-2, scoring='adjusted_mutual_info_score') 
            # grid.fit(xValidate)
            # # get best parameter and update model
            # bestParameters = grid.best_params_
            # self.model = cluster.DBSCAN(eps=bestParameters['eps'],min_samples=bestParameters['min_samples'])
        return bestParameters

    # fit model
    def fit(self, xTrain, yTrain):
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

            

