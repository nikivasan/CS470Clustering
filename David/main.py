## Made by David Gaviria    April 13, 2023 ##

from GenreClusterer import GenreClusterer
import pandas as pd
import argparse
from sklearn import model_selection


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

    # process data table, get genre labels
    data, genres = processData(args.input)
    # Training + Validation Data: xTrain = 63% | xValidate = 27%
    xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(data, genres, train_size=0.7) 
    
    # KMeans
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Running kMeans..')
    print('Finding optimal model parameters...')
    model = GenreClusterer('KMeans')
    bestK, params = model.find_parameters(xValidate)
    print('K-Values vs Silhouette Coefficients:')
    print(params)
    print('Optimal K: %s' %bestK[0])
    print('Silhouette Score: %s' %bestK[1])
    print('Training model with K value: %s' %bestK[0])
    model.update(bestK[0])
    model.fit(xTrain, yTrain)

    # print results
    print('==== Cluster profiles ===')
    model.displayClusters()

    # # DBScan
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Running DBScan..')
    print('Finding optimal model parameters...')
    model = GenreClusterer('DBScan')
    bestEps, params = model.find_parameters(xValidate)
    print('Epsilon vs Silhouette Coefficients:')
    print(params)
    print('Optimal Epsilon: %s' %bestEps[0])
    print('Silhouette Score: %s' %bestEps[1])
    model.update(bestEps[0])
    model.fit(xTrain, yTrain)
    
    # print results
    print('==== Cluster profiles ===')
    model.displayClusters()

    print('Terminated successfully')


if __name__ == '__main__':
    main()