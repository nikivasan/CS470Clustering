This repo is organized into three main folders. DBSCAN and KMEANS hold the plots specific to each model. 

There are two kinds of plots for each model - one is a line graph indicating the optimal number of clusters
with respect to the model's parameters, and the other is a scatterplot visualizing the cluster outputs for each
model on the first two principal components of the input data. 
* Run1 is on the full dataset
* Run2 is on a subset data (genres = Rap/RnB, attributes = tempo, liveness)

The pairwise plots folders are scatterplots demonstrating the relationship between all combinations of attributes 
in the original data, with different subsets of genres. The pre-processing folder contains all the plots relating
to data exploration and pre-processing.