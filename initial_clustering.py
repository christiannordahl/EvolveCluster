import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as spt
import pickle
import os

from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore
from functions import calculate_distances, kmedoids, IC_av, calculate_connectivity

def initial_kmedoids_clustering(data, mini, maxi, iterations, distances=None, centroids=None):
	if distances is None:
		distances = calculate_distances(data.to_numpy(copy=True))

	silhouette_scores, silhouette_clusters = [],[]
	ic_scores, ic_clusters = [],[]
	connectivity_scores, connectivity_clusters = [],[]
	for num_clusters in range(mini, maxi):
		best_silhouette, best_ic, best_connectivity = None, None, None
		sil_cluster, ic_cluster, conn_cluster = None, None, None
		
		print(num_clusters)
		j = 0
		while j < iterations:
			print(j)
			j += 1
			try:
				C, M = kmedoids(distances, num_clusters)
			except Exception as e:
				continue
			clusters = [None for x in range(len(data))]
			for i in range(len(M)):
				for index in C[i]:
					clusters[index] = M[i]
			silhouette = silhouette_score(distances, clusters, metric='precomputed')
			connectivity = calculate_connectivity(data, 
													clusters,
													[x for x in range(data.shape[1])],
													10, distance_matrix=distances)['CONN'].sum()

			if best_silhouette == None or silhouette > best_silhouette:
				best_silhouette = silhouette
				sil_cluster = clusters
			if best_connectivity == None or connectivity < best_connectivity:
				best_connectivity = connectivity
				conn_cluster = clusters

		silhouette_scores.append(best_silhouette)
		silhouette_clusters.append(sil_cluster)
		ic_scores.append(best_ic)
		ic_clusters.append(ic_cluster)
		connectivity_scores.append(best_connectivity)
		connectivity_clusters.append(conn_cluster)

	return [silhouette_scores,
			silhouette_clusters,
			ic_scores,
			ic_clusters,
			connectivity_scores,
			connectivity_clusters]

def delmh():
	files = ["1185", "1186", "1198", "1199", "1200", "1201", "1222", "1223", "1224", "1225"]
	#files = ["1185", "1186"]
	#files = ["1198", "1199"]
	#files = ["1200", "1201"]
	#files = ["1222", "1223"]
	#files = ["1224", "1225"]
	mini = 2
	maxi = 10
	iterations = 100

	for file in files:
		for i in range(6):
			print("Starting", file)
			data = pd.read_csv('data/delmhnew/initial_tests/profilers/segmented/%s_%d.csv'%(file,i), index_col=0, parse_dates=True)
			data = data.apply(zscore, axis=1, result_type='expand')
			distances = pd.DataFrame(calculate_distances(data.to_numpy(copy=True)))

			returns = initial_kmedoids_clustering(data.to_numpy(copy=True), mini, maxi, iterations, distances.to_numpy(copy=True))

			index = [x for x in range(mini, maxi)]
			cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
			df = pd.DataFrame(columns = cols)
			for j in range(6):
				df[cols[j]] = returns[j]
			df.index = index
			df.to_csv('data/delmhnew/results/initial_tests/scores/zscore/%s_%d.csv'%(file,i))

def cover_type():
	mini = 7
	maxi = 8
	iterations = 100
	files = ['data/cover_type/','data/cover_type/']

	for i in range(len(files)):
		data = pd.read_csv(files[i]+'%d.csv'%(i), index_col=0)
		distances = pd.DataFrame(calculate_distances(data.to_numpy(copy=True)))

		returns = initial_kmedoids_clustering(data.to_numpy(), mini, maxi, iterations, distances.to_numpy(copy=True))		
		index = [x for x in range(mini, maxi)]
		cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
		df = pd.DataFrame(columns = cols)
		for j in range(6):
			df[cols[j]] = returns[j]
		df.index = index
		df.to_csv(files[i] + 'scores/%d.csv'%(i))

def synth():
	mini = 5
	maxi = 6
	iterations = 250
	files = ['rbfgenerator/2-dim_0.csv']#, 'rbfgenerator/8-dim_0.csv', 'rbfgenerator/14-dim_0.csv']
	for i in range(len(files)):
		data = pd.read_csv(files[i])
		data = data.iloc[:,:-1]
		distances = pd.DataFrame(calculate_distances(data.to_numpy(copy=True)))

		returns = initial_kmedoids_clustering(data.to_numpy(), mini, maxi, iterations, distances.to_numpy(copy=True))		
		index = [x for x in range(mini, maxi)]
		cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
		df = pd.DataFrame(columns = cols)
		for j in range(6):
			df[cols[j]] = returns[j]
		df.index = index
		df.to_csv(files[i] + 'scores.csv')


def main():
	delmh()
	cover_type()
	synth()

if __name__ == '__main__':
	main()
