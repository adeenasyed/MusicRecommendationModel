# Anita Soroush

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
import os
import csv

def music_recommender(userPreferences):
        raw_data = pd.read_csv("../input training and test sets/cleaned_training_set.csv", dtype={'track': 'str', 'artist': 'str'})
        print(raw_data.shape)
        pd.set_option('display.max_columns', None)
        raw_data.info()

        # data cleaning --------------------------------------------------------------------------------------
        # nulls = raw_data.isnull().sum()
        # print(nulls)

        training_data = raw_data.drop(['id', 'track', 'artist'], axis=1, inplace=False)

        # print(training_data.shape)
        # training_data = training_data[training_data.key != -1]
        # print("after dropping some rows:\n", training_data.shape)
        # print(training_data.head())

        # print(training_data.shape)
        # print(training_data.duplicated().any())
        # training_data.drop_duplicates(inplace=True)
        # print(training_data.shape)

        training_data.hist()
        plt.show()

        # this global scalar will be fitted on training data and will be used for both training and test data
        global_scalar = MinMaxScaler()
        global_scalar.fit(training_data)
        training_data = pd.DataFrame(global_scalar.transform(training_data), index=training_data.index, columns=training_data.columns)

        pca = PCA().fit(training_data)
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

        plt.figure()
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA: Cumulative Explained Variance vs Number of Components')
        plt.show()

        pca_optimal = PCA(n_components=0.9) # retain 90% of variance based on elbow plot
        pca_optimal.fit(training_data)
        training_data = pca_optimal.transform(training_data)
        print(f"Reduced to {pca_optimal.n_components_} dimensions")
        training_data = pd.DataFrame(training_data)

        training_data = training_data.join(raw_data[['id', 'track', 'artist']])

        training_data.hist()
        plt.show()

        training_data.info()

        # clustering ----------------------------------------------------------------------------------------
        wcss = []
        for i in range(1, 20):
            kmeans = KMeans(i)
            kmeans.fit(training_data.drop(['id', 'track', 'artist'], axis=1, inplace=False))
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)

        number_clusters = range(1, 20)
        plt.plot(number_clusters, wcss)
        plt.title('The Elbow title')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.show()

        knee = KneeLocator(range(1, 20), wcss, curve='convex', direction='decreasing')
        optimal_clusters = knee.elbow
        print(f"Optimal number of clusters: {optimal_clusters}")

        kmeans = KMeans(n_clusters=optimal_clusters)
        training_data_clustered = kmeans.fit(training_data.drop(['id', 'track', 'artist'], axis=1, inplace=False))
        training_data["cluster"] = training_data_clustered.labels_
        centroids = training_data_clustered.cluster_centers_
        print(training_data.head())

        # making output...........................................................................................
        userPreferences.drop(userPreferences.columns.difference(["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]), 1, inplace=True)

        # input normalizing
        userPreferences = pd.DataFrame(global_scalar.transform(userPreferences), index=userPreferences.index, columns=userPreferences.columns)
        userPreferences = pd.DataFrame(pca_optimal.transform(userPreferences))

        fields = ["id", "track", "artist", "cluster"]

        # single playlist
        single_playlist = []
        for i in range(5):
            cluster_index = (training_data_clustered.predict(userPreferences.iloc[[i]]))[0]
            print(cluster_index)
            cluster_songs = training_data[training_data.cluster == cluster_index]
            cluster_songs.drop(cluster_songs.columns.difference(["id", "track", "artist", "cluster"]), 1, inplace=True)
            single_playlist.append((cluster_songs.sample()).values.flatten().tolist())
            print(single_playlist[i])

        filename = "single_playlist.csv"

        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(fields)

            # writing the data rows
            csvwriter.writerows(single_playlist)

        # 5 playlists
        for i in range(5):
            ith_playlist = []
            filename = "pl" + str(i + 1) + ".csv"
            cluster_index = (training_data_clustered.predict(userPreferences.iloc[[i]]))[0]
            cluster_songs = training_data[training_data.cluster == cluster_index]
            cluster_songs.drop(cluster_songs.columns.difference(["id", "track", "artist", "cluster"]), 1, inplace=True)
            for j in range(5):
                ith_playlist.append((cluster_songs.sample()).values.flatten().tolist())

            with open(filename, 'w') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)

                # writing the fields
                csvwriter.writerow(fields)

                # writing the data rows
                csvwriter.writerows(ith_playlist)


def main(args) -> None:
    """ Main function to be called when the script is run from the command line. 
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.
    
    Parameters
    ----------
    args: list 
        list of arguments from the command line (here is just the path of a file like input_tracks.csv)
    Returns
    -------
    None
    """
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        if not os.path.isfile(file_name):
            print("File does not exist")
            sys.exit()
        else:
            userPreferences = pd.read_csv(file_name)
            music_recommender(userPreferences)

if __name__ == "__main__":
    """get arguments from command line
    you just have to write the name of the file that contains the users favorite tracks.
    these tracks are now in input_tracks.csv """
    args = sys.argv
    main(args)