## Made by David Gaviria    April 13, 2023 ##
## Edits and Add Ons by Niki Vasan ##

from GenreClusterer import GenreClusterer
import numpy as np
import pandas as pd
import argparse
from sklearn import model_selection
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


# function to process data
def processData(filename):
    data = pd.read_csv(filename)
    genres = data['genre']
    data.drop(labels=['genre','duration_ms','time_signature'], axis=1, inplace=True)   
    return data, genres


# main
def main():
    # get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        help='Name of input file')
    args = parser.parse_args()
    
    # ++++++++ OPTIONAL: Test Subsets of Data ++++++++ #
    genre_options = ['Rap', 'RnB']

    data = pd.read_csv(args.input)
    data = data[data['genre'].isin(genre_options)]
    genres = data['genre']
    data = data[['tempo', 'liveness']]

    # ++++++++++++++++++++++++++++++++++++++++++++++++ #
    
    # # process data table, get genre labels
    # data, genres = processData(args.input)

    # Training + Validation Data: xTrain = 63% | xValidate = 27%
    xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(data, genres, train_size=0.9) 
    
    # # KMeans
    # print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print('Running kMeans..')
    # print('Finding optimal model parameters...')
    # model = GenreClusterer('KMeans')
    # # find optimal parameters
    # bestK, params = model.find_parameters(xValidate)
    # print('K-Values vs Silhouette Coefficients:')
    # print(params)
    # print('Optimal K: %s' %bestK[0])
    # print('Training model with K value: %s' %bestK[0])
    #  # cluster
    # model.update(bestK[0])
    # yHat = model.fit(xTrain)

    # # visualize PCA plot 
    # model.plot_components(xTrain)

    # # print results
    # print('==== Cluster profiles ===')
    # model.generateClusterProfiles(yHat, yTrain)
    # model.displayClusters()

    # # DBScan
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Running DBScan..')
    print('Finding optimal model parameters...')
    model = GenreClusterer('DBScan')
    # find optimal parameters
    bestEps, params = model.find_parameters(xValidate)
    print('Epsilon vs Silhouette Coefficients:')
    print(params)
    print('Optimal Epsilon: %s' %bestEps[0])

    
    # cluster
    model.update(bestEps[0])
    yHat = model.fit(xTrain)
    print("Y HAT: ", yHat)
    print('Silhouette Score: %.4f' % metrics.silhouette_score(xTrain, yHat))

    # visualize PCA plot
    model.plot_components(xTrain)
    
    # # print results
    # print('==== Cluster profiles ===')
    # model.displayClusters()

    print('Terminated successfully')


if __name__ == '__main__':
    main()