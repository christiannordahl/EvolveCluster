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
from functions import IC_av
from matplotlib.patches import ConnectionPatch

class EvolveCluster(object):
	def __init__(self, data, clusters, centroids, distances, tau, threshold=0.3):
		"""
		Requires data to be initially clustered beforehand.
		"""
		self.data = [data]
		self.clusters = [clusters]
		self.centroids = [centroids]
		self.distances = [distances]
		self.threshold = threshold
		self.nr_increments = 0
		self.transitions = []
		self.tau = tau

	def to_csv_delmh(self, path):
		for i in range(len(self.data)):
			df = pd.DataFrame(self.data[i])
			clusters = [None for j in range(len(self.data[i]))]
			cluster_ids = [None for j in range(len(self.data[i]))]
			for j in range(len(self.centroids[i])):
				for k in range(len(self.clusters[i][j])):
					clusters[self.clusters[i][j][k]] = self.centroids[i][j]
					cluster_ids[self.clusters[i][j][k]] = j
			df['cluster'] = clusters
			df['cluster_ids'] = cluster_ids
			df.to_csv(os.path.join(path, '%d.csv'%(i)), index=None)

	def to_csv_new(self, data_path, result_path):
		print(self.transitions)
		total_df = None
		for i in range(len(self.data)):
			df = pd.read_csv(os.path.join(data_path, '%d.csv'%(i)))
			clusters = [None for j in range(len(self.data[i]))]
			cluster_ids = [None for j in range(len(self.data[i]))]
			print(i)
			for j in range(len(self.centroids[i])):
				if i == 0:
					j = str(j)
				for k in range(len(self.clusters[i][j])):
					jj=int(j)
					clusters[self.clusters[i][j][k]] = self.centroids[i][jj]
					cluster_ids[self.clusters[i][j][k]] = jj

			df['actual_cluster'] = df['cluster']
			df['cluster'] = clusters
			df['cluster_ids'] = cluster_ids

			if total_df is None:
				total_df = df.copy()
			else:
				total_df = total_df.append(df, ignore_index=True)
			df.to_csv(os.path.join(result_path,'%d.csv'%(i)))

		with open(os.path.join(result_path, 'transitions.txt'), 'w') as f:
			for i in range(len(self.transitions)):
				f.write(json.dumps(self.transitions[i])+'\n')
			f.close()
		total_df.to_csv(os.path.join(result_path, 'total.csv'))

	def to_csv_new_synthetic(self, path, dimension, seed_dim):
		print(self.transitions)
		total_df = None
		for i in range(len(self.data)):
			df = pd.read_csv(os.path.join('data/', path, '%s-dim_%d.csv'%(seed_dim,i)))
			clusters = [None for j in range(len(self.data[i]))]
			cluster_ids = [None for j in range(len(self.data[i]))]
			for j in range(len(self.centroids[i])):
				if i == 0:
					j = str(j)
				for k in range(len(self.clusters[i][j])):
					jj=int(j)
					clusters[self.clusters[i][j][k]] = self.centroids[i][jj]
					cluster_ids[self.clusters[i][j][k]] = jj

			df['actual_cluster'] = df['cluster']
			df['cluster'] = clusters
			df['cluster_ids'] = cluster_ids

			if total_df is None:
				total_df = df.copy()
			else:
				total_df = total_df.append(df, ignore_index=True)
			df.to_csv(os.path.join('results', path,'%s-dim_%d.csv'%(seed_dim,i)))

		with open(os.path.join('data/', path, 'transitions.txt'), 'w') as f:
			for i in range(len(self.transitions)):
				f.write(json.dumps(self.transitions[i])+'\n')
			f.close()
		total_df.to_csv(os.path.join('results', path,'%s_dim_total.csv'%(seed_dim)))

	def to_csv_new_cover(self, path):
		total_df = None
		for i in range(len(self.data)):
			df = pd.read_csv(os.path.join(path, '%d.csv'%(i)))
			clusters = [None for j in range(len(self.data[i]))]
			cluster_ids = [None for j in range(len(self.data[i]))]
			print(i)
			for j in range(len(self.centroids[i])):
				if i == 0:
					j = str(j)
				for k in range(len(self.clusters[i][j])):
					jj=int(j)
					clusters[self.clusters[i][j][k]] = self.centroids[i][jj]
					cluster_ids[self.clusters[i][j][k]] = jj

			df['actual_cluster'] = df['cluster']
			df['cluster'] = clusters
			df['cluster_ids'] = cluster_ids

			if total_df is None:
				total_df = df.copy()
			else:
				total_df = total_df.append(df, ignore_index=True)
			df.to_csv(os.path.join(path,'%d.csv'%(i)))

		with open(os.path.join(path, 'transitions.txt'), 'w') as f:
			for i in range(len(self.transitions)):
				f.write(json.dumps(self.transitions[i])+'\n')
			f.close()
		total_df.to_csv(os.path.join(path,'total.csv'))

	def homogenity(self, mean=True):
		clusters = [None for x in range(len(self.data[self.nr_increments]))]
		for i in range(len(self.centroids[self.nr_increments])):
			for index in self.clusters[self.nr_increments][i]:
				clusters[index] = self.centroids[self.nr_increments][i]

		sil_scores = silhouette_samples(self.distances[self.nr_increments], clusters, metric='precomputed')
		
		div_scores = [[] for x in range(len(self.centroids[self.nr_increments]))]
		for key in self.clusters[self.nr_increments]:
			for index in self.clusters[self.nr_increments][key]:
				div_scores[key].append(sil_scores[index])

		if mean == True:
			for i in range(len(div_scores)):
				div_scores[i] = np.mean(np.array(div_scores[i]))

		return div_scores

	def split(self):
		if(len(self.centroids[self.nr_increments]) > 1):
			total_hom_score = self.homogenity(False)
			total_hom_score = np.mean([y for x in total_hom_score for y in x])
			homogenity_scores = self.homogenity()
			indices_low_scores = sorted(range(len(homogenity_scores)), key = lambda sub: homogenity_scores[sub])[:]

			i = 0
			while i < len(indices_low_scores):
				if len(self.clusters[self.nr_increments][indices_low_scores[i]]) > 1:
					old_clusters = copy.deepcopy(self.clusters[self.nr_increments])
					old_centroids = copy.deepcopy(self.centroids[self.nr_increments])

					distances = self.distances[self.nr_increments]
					indices = self.clusters[self.nr_increments][indices_low_scores[i]]
					mask = [x in indices for x in range(len(self.data[self.nr_increments]))]
					mask2 = np.array([[x & y for x in mask] for y in mask])
					index_converter = [x for x in range(len(mask)) if mask[x] is True]

					distances = distances[mask2]
					distances = np.reshape(distances, (-2,int(len(distances)**0.5)))
					centroids = np.where(distances == np.amax(distances))
					centroids = [centroids[0][0],centroids[-1][0]]

					C, new_centroids = kmedoids(distances, 2, centroids)

					cluster_indices = np.array([index_converter[x] for x in C[0]])
					self.clusters[self.nr_increments][indices_low_scores[i]] = cluster_indices
					cluster_indices = [index_converter[x] for x in C[1]]
					self.clusters[self.nr_increments][len(self.clusters[self.nr_increments])] = cluster_indices
					self.centroids[self.nr_increments][indices_low_scores[i]] = index_converter[new_centroids[0]]
					self.centroids[self.nr_increments].append(index_converter[new_centroids[1]])

					C, centroids = kmedoids(self.distances[self.nr_increments], 
											len(self.centroids[self.nr_increments]), 
											self.centroids[self.nr_increments])
					self.clusters[self.nr_increments] = C
					self.centroids[self.nr_increments] = centroids

					new_homogenity_scores = self.homogenity(False)
					new_hom_score = np.mean([y for x in new_homogenity_scores for y in x])

					if new_hom_score > (total_hom_score + abs(total_hom_score*self.tau)):
						self.transitions[self.nr_increments-1][indices_low_scores[i]] = [indices_low_scores[i], len(self.clusters[self.nr_increments])-1]
						return True

					self.clusters[self.nr_increments] = old_clusters
					self.centroids[self.nr_increments] = old_centroids
				i += 1

		return False

	def cluster(self, data):
		for seg in data:
			self.cluster_method(seg)
			while(self.split()):
				pass

	def cluster_method(self, data):
		"""
		Requires self-implementation depending on what clustering method
		you choose. Receives data as a parameter and has to return a list
		containing a list of corresponding cluster values per element and 
		a list of centroids. Below is an implementation using k-medoids.
		"""
		
		centroids = self.centroids[self.nr_increments]
		num_clusters = len(centroids)
		for centroid in centroids:
			data = np.concatenate((data, [self.data[self.nr_increments][centroid]]))#.iloc[centroid,:].to_numpy(copy=True)]))
		centroids = [x for x in range(len(data)-num_clusters, len(data))]

		D = calculate_distances(data)
		data = data[:-num_clusters]
		C = {}
		C, D, centroids = self.initial_partiton(C, D, centroids, num_clusters)

		transitions = {}
		j = 0
		for i in range(num_clusters):
			transitions[i] = None
			if centroids[i] is not None:
				transitions[i] = [j]
				j += 1
		for i in range(num_clusters-1, -1, -1):
			if centroids[i] == None:
				del centroids[i]
				num_clusters -= 1
		self.transitions.append(transitions)

		C, centroids = kmedoids(D, num_clusters, centroids)
		
		self.distances.append(D)
		self.centroids.append(centroids)
		self.clusters.append(C)
		self.data.append(data)
		self.nr_increments += 1

	def initial_partiton(self, C, D, medoids, num_clusters):
		J = np.argmin(D[:,medoids], axis=1)
		for kappa in range(num_clusters):
			C[kappa] = np.where(J==kappa)[0]

		D = D[:-num_clusters,:-num_clusters]
		for kappa in range(num_clusters):
			for i in range(len(C[kappa])):
				if C[kappa][i] == medoids[kappa]:
					C[kappa] = np.delete(C[kappa], i)

		for kappa in range(num_clusters):
			J = np.mean(D[np.ix_(C[kappa],C[kappa])], axis=1)
			if J.size == 0:
				del C[kappa]
				medoids[kappa] = None
			else:
				j = np.argmin(J)
				medoids[kappa] = C[kappa][j]

		return C, D, medoids

def cover_test(num_clusters, tau, initial_centroids=None):	
	data = pd.read_csv('data/cover_type/%d.csv'%(0), index_col=0)
	tmp = data.iloc[:,:14].to_numpy()
	D = calculate_distances(tmp)
	tmp = None
	
	with open('data/cover_type/initial_clusters.json','r') as f:
		C = json.load(f)
	medoids = pd.read_csv('data/cover_type/medoids.csv',index_col=None)['medoids'].to_list()

	a = EvolveCluster(data.iloc[:,:14].to_numpy(), C, medoids, D, tau)
	new_data = []
	for i in range(1,2):
		new_data.append(pd.read_csv('data/cover_type/%d.csv'%(i)).iloc[:,:14].to_numpy())

	a.cluster(new_data)
	pickle.dump(a, open('results/covertype/results.pkl', 'wb'), protocol=4)
	a.to_csv_new_cover('results/covertype/')

def s1(dataset, length, tau):
	data = pd.read_csv('data/s1/%s/%d.csv'%(dataset, 0))

	with open('data/s1/%s/%d.txt'%(dataset, 0), 'r') as f:
		a = f.readlines()
		f.close()
	a = [int(x) for x in a[2].rstrip().split(',')]
	medoids = sorted(list(set(a)))
	C = {}
	for i in range(len(medoids)):
		C[str(i)] = []
		for j in range(len(a)):
			if a[j] == medoids[i]:
				C[str(i)].append(j)
		C[str(i)] = np.array(C[str(i)])

	num_clusters = len(medoids)
	tmp = data.iloc[:,:2].to_numpy(copy=True)

	D = calculate_distances(tmp)
	a = EvolveCluster(data.iloc[:,:2].to_numpy(copy=True), C, medoids, D, tau)

	new_data = []
	for i in range(1,length):
		new_data.append(pd.read_csv('data/s1/%s/%d.csv'%(dataset, i)).iloc[:,:2].to_numpy(copy=True))

	a.cluster(new_data)
	pickle.dump(a, open('results/s1/%s/results.pkl'%(dataset), 'wb'))
	a.to_csv_new('data/s1/%s'%(dataset), 'results/s1/%s'%(dataset))

def delmh(tau):
	# 1200 = 6, 1222 = 3.
	data_path = 'data/delmh/data'
	houses = ['1200','1222']
	rows = {'1200':6,'1222':3}
	for house in houses:
		info = pd.read_csv(data_path+'/initial_clusters/'+house+'_0.csv', index_col=0, parse_dates=True)
		clusters = json.loads(info.loc[rows[house]]['sil_clusters'])
		medoids = sorted(list(set(clusters)))
		data = [pd.read_csv(data_path+'/'+house+'_%d.csv'%(x), index_col=0, parse_dates=True) for x in range(1)]
		for i in range(1,5):
			data.append(pd.read_csv(data_path+'/'+house+'_%d.csv'%(i), index_col=0, parse_dates=True).iloc[:,:24])
		C = {}
		for i in range(len(medoids)):
			C[i] = []
			for j in range(len(clusters)):
				if clusters[j] == medoids[i]:
					C[i].append(j)

		tmp = data[0].iloc[:,:24].to_numpy()
		D = calculate_distances(tmp)
		a = EvolveCluster(data[0].iloc[:,:24].to_numpy(), C, medoids, D, tau)
		a.cluster(data[1:])
		pickle.dump(a, open('results/delmh/%s/normal/EvolveCluster/results.pkl'%(house), 'wb'))
		a.to_csv_delmh('results/delmh/%s/normal/EvolveCluster'%(house))


def synthetic(dimension, length, seed_dim, tau):
	data = pd.read_csv('data/rbfgenerator/%s-dim_%d.csv'%(seed_dim, 0))
	data['clusters'] = np.nan
	tmp = data.iloc[:,:dimension].to_numpy(copy=True)

	D = calculate_distances(tmp)
	medoids = []
	for i in range(5):
		test = data[data['cluster'] == i]
		D_test = calculate_distances(test.iloc[:,:dimension].to_numpy(copy=True))
		medoid = np.argmin(D_test.sum(axis=0))
		medoid = int(test.iloc[medoid].name)
		data.loc[data['cluster'] == i, 'clusters'] = medoid
		medoids.append(medoid)

	a = data['clusters'].to_list()
	C = {}
	for i in range(len(medoids)):
		C[str(i)] = []
		for j in range(len(a)):
			if a[j] == medoids[i]:
				C[str(i)].append(j)
		C[str(i)] = np.array(C[str(i)])

	a = EvolveCluster(data.iloc[:,:dimension].to_numpy(copy=True), C, medoids, D, tau)

	new_data = []
	for i in range(1,length):
		new_data.append(pd.read_csv('data/rbfgenerator/%s-dim_%d.csv'%(seed_dim, i)).iloc[:,:dimension].to_numpy(copy=True))

	a.cluster(new_data)
	pickle.dump(a, open('results/rbfgenerator/%s-dim_results.pkl'%(seed_dim), 'wb'))
	a.to_csv_new_synthetic('rbfgenerator/',dimension, seed_dim)

def new_experiment():
	print('Starting continuous')
	s1('continuous', 5, 0.08)
	print('Starting original')
	s1('original', 5, 0.08)
	print('Starting cover_type')
	cover_test(7, 0.30)
	#print('Starting delmh')
	#delmh(0.13)	# 3 = third row, which is k=4	# -6, aka 1200, with 3rd row, aka k=4 is fun

	print('Starting RBFGenerated dataset')
	taus = [0.09, 0.02, 0.15, 0.05]
	seeds = [1,2,6,29]
	for seed,tau in zip(seeds, taus):
		synthetic(2,5, "%d_%d"%(seed, 2), tau)
		synthetic_indiv_metrics(2, "%d_%d"%(seed,2))
def main():
	new_experiment()

if __name__ == '__main__':
	main()
