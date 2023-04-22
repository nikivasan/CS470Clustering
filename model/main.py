## Made by David Gaviria and Niki Vasan   April 13, 2023 ##

from GenreClusterer import GenreClusterer
import pandas as pd
import argparse
from sklearn import model_selection
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from pyclustertend import hopkins
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


# function to process data
def processData(filename):
    data = pd.read_csv(filename)
    genres = data['genre']
    data.drop(labels=['genre','duration_ms','time_signature'], axis=1, inplace=True)   
    return data, genres

def hopkinsStat(data):
    hopkins_stat = hopkins(data, data.shape[0])
    return round(hopkins_stat,5)

# function to visualize data
def plotData(data, genres):
    # get components
    pca = PCA(n_components=2)
    xTrainPCA = pca.fit_transform(data)
    df = pd.concat([data.reset_index(drop=True), pd.DataFrame(xTrainPCA)], axis=1)
    df['Labels'] = genres
    df = df.rename(columns={0:'Component 1', 1:'Component 2'})
    
    # plot data
    x_axis = df['Component 1']
    y_axis = df['Component 2']
    color = df['Labels']

    plt.figure(figsize=(10,8))
    sns.scatterplot(x_axis, y_axis, hue=color )
    plt.title('PCA: Genres')
    plt.show()


# main
def main():
    # get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        help='Name of input file')
    args = parser.parse_args()
    
    # ++++++++ OPTIONAL: Test Subsets of Data ++++++++ #
    # genre_options = ['Rap', 'RnB']

    # data = pd.read_csv(args.input)
    # data = data[data['genre'].isin(genre_options)]
    # genres = data['genre']
    # data = data[['tempo', 'liveness']]

    # ++++++++++++++++++++++++++++++++++++++++++++++++ #
    
    # process data table, get genre labels
    data, genres = processData(args.input)

    # check clustering tendency using hopkin's statistic
    print('Hopkin\'s Statistic: ', hopkinsStat(data))

    # visualize data
    plotData(data, genres)

    # Training + Validation Data: xTrain = 70% | xValidate = 30%
    xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(data, genres, train_size=0.7) 
    
    # KMeans
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Running kMeans..')
    print('Finding optimal model parameters...')
    model = GenreClusterer('KMeans')
    # find optimal parameters
    bestK, params = model.find_parameters(xValidate)
    print('K-Values vs Silhouette Coefficients:')
    print(params)
    print('Optimal K: %s' %bestK[0])
    print('Training model with K value: %s' %bestK[0])
     # cluster
    model.update(bestK[0])
    yHat = model.fit(xTrain)

    # visualize PCA plot 
    model.plot_components(xTrain)

    # print results
    print('==== Cluster profiles ===')
    model.generateClusterProfiles(yHat, yTrain)
    model.displayClusters()

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
    
    # print results
    print('==== Cluster profiles ===')
    model.displayClusters()

    print('Terminated successfully')


if __name__ == '__main__':
    main()