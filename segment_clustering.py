import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as spt
import pickle
import os
import json
import sys
import copy

from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore
from functions import calculate_distances
from functions import kmedoids
from matplotlib.patches import ConnectionPatch


def kmed(filepath, num_clusters, savepath, initial_centroids=None, limit=100):
	data = pd.read_csv(filepath)


	tmp = data.iloc[:,:2].to_numpy(copy=True)
	D = calculate_distances(tmp)

	best_silhouette = None
	best_clusters = None
	for i in range(limit):
		C, medoids = kmedoids(D, num_clusters, initial_centroids)

		clusters = [None for x in range(len(data))]
		for i in range(len(medoids)):
			for index in C[i]:
				clusters[index] = medoids[i]

		silhouette = silhouette_score(D, clusters, metric='precomputed')
		if best_silhouette == None or silhouette > best_silhouette:
			best_silhouette = silhouette
			best_clusters = clusters
			best_medoids = medoids

	data = []
	data.append(str(best_silhouette))
	data.append(",".join([str(x) for x in initial_centroids]))
	data.append(",".join([str(x) for x in best_clusters]))

	with open(savepath,'w') as f:
		for i in range(3):
			f.write(data[i]+'\n')
		f.close()


def main():
	files = ['data/s1new/original/%d.csv'%(x) for x in range(5)]+['data/s1new/continuous/%d.csv'%(x) for x in range(5)]
	savefiles = ['data/s1new/original/%d.txt'%(x) for x in range(5)]+['data/s1new/continuous/%d.txt'%(x) for x in range(5)]
	initial_centroids = []
	for i in range(10):
		centroids = []
		df = pd.read_csv(files[i])
		clusters = df['cluster'].unique().tolist()
		for cluster in clusters:
			centroids.append(df[df['cluster'] == cluster].first_valid_index())
		initial_centroids.append(centroids)
	num_clusters = [len(x) for x in initial_centroids]

	for i in range(10):
		kmed(files[i], num_clusters[i], savefiles[i], initial_centroids[i], 100)

if __name__ == '__main__':
	main()
