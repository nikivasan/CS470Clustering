## Made by David Gaviria    April 13, 2023 ##

from GenreClusterer import GenreClusterer
import pandas as pd
import argparse
from sklearn import model_selection


# function to process data
def processData(filename):
    data = pd.read_csv(filename)
    genres = data['genre']
    data.drop(labels=['genre','duration_ms','time_signature'], axis=1, inplace=True)         # needs to be left or reconverted to categorical
    return data, genres


# main
def main():
    # get inputs
    # parser = argparse.ArgumentParser()
    # parser.add_argument(input,
    #                     help='Name of input file')
    # args = parser.parse_args()

    argsinput = r"C:\Users\david\Documents\GitHub\CS470FinalProject\David\temp_music_scaled.csv"  # fake input
    dataPath = ''
    # process data table, get genre labels
    data, genres = processData(argsinput)
    xTrainAndVal, xTest, yTrainAndVal, yTest = model_selection.train_test_split(data, genres, train_size=0.9)
    xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(xTrainAndVal, yTrainAndVal, train_size=0.7)
    
    # KMeans
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Running kMeans..')
    model = GenreClusterer('KMeans')
    bestK = model.find_parameters(xValidate, yValidate, v=True)
    print('Optimal parameters: %s' % bestK)
    model.fit(xTrain, yTrain)
    # print results
    print('==== Cluster profiles ===')
    model.displayClusters()

    # DBScan
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Running DBScan..')
    model = GenreClusterer('DBScan')
    bestK = model.find_parameters(xValidate, yValidate, v=True)
    print('Optimal parameters: %s' % bestK)
    model.fit(xTrain, yTrain)
    # print results
    print('==== Cluster profiles ===')
    model.displayClusters()

    # recommender system
    ###
    print('Terminated successfully')




if __name__ == '__main__':
    main()