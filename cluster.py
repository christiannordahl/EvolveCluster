import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as spt
import pickle
import os
import json

from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore
from functions import calculate_distances
from functions import kmedoids
from matplotlib.patches import ConnectionPatch

class WTS(object):
	def __init__(self, data, clusters, centroids, distances, threshold):
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

	def to_csv(self, file):
		for i in range(len(self.data)):
			df = pd.DataFrame(self.data[i])
			clusters = [None for j in range(len(self.data[i]))]
			for j in range(len(self.centroids[i])):
				for k in range(len(self.clusters[i][j])):
					clusters[self.clusters[i][j][k]] = self.centroids[i][j]
			df['cluster'] = clusters
			df.to_csv(file, index=None)


	def plot_delmh(self, path, file):
		colors=['teal', 'aqua', 'pink', 'brown', 'fuchsia', 'gold', 'lime', 'orange', 'purple', 'red', 'yellow', 'khaki', 'grey', 'tan', 'coral']
		max_clusts = max([len(x) for x in self.clusters])
		fig, axs = plt.subplots(max_clusts, self.nr_increments+1, sharex=True, sharey=True,figsize=(24,6))

		for i in range(self.nr_increments+1):
			for j in range(len(self.clusters[i])):
				for index in self.clusters[i][j]:
					axs[j][i].plot(self.data[i][index], color=colors[j], alpha=0.1)
				axs[j][i].plot(self.data[i][self.centroids[i][j]], color='black')

		for i in range(self.nr_increments):
			for key in self.transitions[i]:
				if self.transitions[i][key] is None:
					continue

				for target in self.transitions[i][key]:
					con = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data", 
	                      axesA=axs[key][i], axesB=axs[target][i+1], arrowstyle="->", color="crimson", lw=3)
					axs[key][i].add_artist(con)


		fig.savefig('%s%s.png'%(path,file))
		plt.close()


	def plot(self):
		colors=['teal', 'aqua', 'pink', 'brown', 'fuchsia', 'gold', 'lime', 'orange', 'purple', 'red', 'yellow', 'khaki', 'grey', 'tan', 'coral']
		fig, axs = plt.subplots(1,self.nr_increments+1,figsize=(24,3), sharex=True, sharey=True)
		for i in range(self.nr_increments+1):
			for key in self.clusters[i]:
				for index in self.clusters[i][key]:
					axs[i].scatter(self.data[i][index][0], self.data[i][index][1], color=colors[key], s=30)
				axs[i].scatter(self.data[i][self.centroids[i][key]][0], self.data[i][self.centroids[i][key]][1], color=colors[key],edgecolors='black', linewidth=1, s=30)

			#plt.ylim(-2,13)
			#plt.xlim(-2,13)
		fig.savefig('test.png')
		plt.close()

	def plot_vs_true(self, true_data, filename):
		colors=['teal', 'aqua', 'pink', 'brown', 'fuchsia', 'gold', 'lime', 'orange', 'purple', 'red', 'yellow', 'khaki', 'grey', 'tan', 'coral']
		fig, axs = plt.subplots(2,self.nr_increments+1,figsize=(24,3), sharex=True, sharey=True)
		
		for i in range(self.nr_increments+1):
			for key in self.clusters[i]:
				for index in self.clusters[i][key]:
					axs[0][i].scatter(self.data[i][index][0], self.data[i][index][1], color=colors[key], s=10)
				axs[0][i].scatter(self.data[i][self.centroids[i][key]][0], self.data[i][self.centroids[i][key]][1], color=colors[key],edgecolors='black', linewidth=3, s=10)

		
			for j in range(len(true_data[i])):
				#print(i,j)
				#print(true_data[0].iloc[j,0])
				#print(true_data[0].iloc[j,1])
				#print(true_data[0].iloc[j,2])
				axs[1][i].scatter(true_data[i].iloc[j,0], true_data[i].iloc[j,1], color=colors[true_data[i].iloc[j,2]-1], s=10)

			#plt.ylim(-2,13)
			#plt.xlim(-2,13)
		fig.savefig(filename)
		plt.close()

	def plot_gray(self):
		colors = ['r', 'g', 'y']
		for i in range(self.nr_increments+1):
			for key in self.clusters[i]:
				for index in self.clusters[i][key]:
					plt.scatter(self.data[i][index][0], self.data[i][index][1], s=150, color='gray')

			plt.ylim(-2,13)
			plt.xlim(-2,13)
			plt.savefig('figures/%d_greyed.png'%(i))
			plt.close()

	def homogenity_silhouette(self):
		clusters = [None for x in range(len(self.data[self.nr_increments]))]
		for i in range(len(self.centroids[self.nr_increments])):
			for index in self.clusters[self.nr_increments][i]:
				clusters[index] = self.centroids[self.nr_increments][i]

		sil_scores = silhouette_samples(self.distances[self.nr_increments], clusters, metric='precomputed')
		
		div_scores = [[] for x in range(len(self.centroids[self.nr_increments]))]
		for key in self.clusters[self.nr_increments]:
			for index in self.clusters[self.nr_increments][key]:
				div_scores[key].append(sil_scores[index])

		for i in range(len(div_scores)):
			div_scores[i] = np.mean(np.array(div_scores[i]))

		return div_scores

	def homogenity_other(self):
		clusters = [None for x in range(len(self.data[self.nr_increments]))]
		for i in range(len(self.centroids[self.nr_increments])):
			for index in self.clusters[self.nr_increments][i]:
				clusters[index] = self.centroids[self.nr_increments][i]

		sil_scores = silhouette_samples(self.distances[self.nr_increments], clusters, metric='precomputed')
		
		div_scores = [[] for x in range(len(self.centroids[self.nr_increments]))]
		for key in self.clusters[self.nr_increments]:
			for index in self.clusters[self.nr_increments][key]:
				div_scores[key].append(sil_scores[index])

		for i in range(len(div_scores)):
			div_scores[i] = np.mean(np.array(div_scores[i]))

		return div_scores


	def split(self):
		if(len(self.centroids[self.nr_increments]) > 1):
			homogenity_scores = self.homogenity_silhouette()
			indices_low_scores = sorted(range(len(homogenity_scores)), key = lambda sub: homogenity_scores[sub])[:]

			for i in range(len(homogenity_scores)):
				#print(self.clusters[self.nr_increments], self.centroids[self.nr_increments])
				if homogenity_scores[indices_low_scores[i]] < self.threshold:
					distances = self.distances[self.nr_increments]
					indices = self.clusters[self.nr_increments][indices_low_scores[i]]
					mask = [x in indices for x in range(len(self.data[self.nr_increments]))]
					mask2 = np.array([[x & y for x in mask] for y in mask])
					index_converter = [x for x in range(len(mask)) if mask[x] is True]

					distances = distances[mask2]
					distances = np.reshape(distances, (-2,int(len(distances)**0.5)))
					centroids = np.where(distances == np.amax(distances))
					centroids = [centroids[0][0],centroids[-1][0]]
					#print(centroids)
					C, new_centroids = kmedoids(distances, 2, centroids)

					cluster_indices = np.array([index_converter[x] for x in C[0]])
					self.clusters[self.nr_increments][indices_low_scores[i]] = cluster_indices
					cluster_indices = [index_converter[x] for x in C[1]]
					self.clusters[self.nr_increments][len(self.clusters[self.nr_increments])] = cluster_indices
					self.transitions[self.nr_increments-1][indices_low_scores[i]] = [indices_low_scores[i], len(self.clusters[self.nr_increments])-1]
					self.centroids[self.nr_increments][indices_low_scores[i]] = index_converter[new_centroids[0]]
					self.centroids[self.nr_increments].append(index_converter[new_centroids[1]])

					return True
		return False



	def cluster(self, data):
		for seg in data:
			self.cluster_method(seg)
			
			while(self.split()):
				C, centroids = kmedoids(self.distances[self.nr_increments], 
										len(self.centroids[self.nr_increments]), 
										self.centroids[self.nr_increments])
				self.clusters[self.nr_increments] = C
				self.centroids[self.nr_increments] = centroids

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

		# Calculate distance and store in D
		D = calculate_distances(data)
		data = data[:-num_clusters]
		# Initial partitioning
		C = {}
		C, D, centroids = self.initial_partiton(C, D, centroids, num_clusters)

		# Loop through centroids and fix the None values, i.e. mark clusters dead and store somehow?
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

		# Remove the initial centroids from matrix
		D = D[:-num_clusters,:-num_clusters]
		# Remove initial centroids from clusters
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

def propertest():
	data = []
	for i in range(6):
		data.append(pd.read_csv('data/s1/continuous_behaviors_%d.csv'%(i), header=None))

	tmp = data[0].iloc[:,:2].to_numpy(copy=True)
	D = calculate_distances(tmp)

	C, medoids = kmedoids(D, 15)

	a = WTS(data[0].iloc[:,:2].to_numpy(copy=True), C, medoids, D, -1)
	new_data = []
	for i in range(1,6):
		new_data.append(data[i].iloc[:,:2].to_numpy(copy=True))

	a.cluster(new_data)

	pickle.dump(a, open("test.pkl", "wb"))
	a = pickle.load(open("test.pkl", "rb"))
	#a.plot()

	a.plot_vs_true(data, "test.png")


def propertest2():
	data = []
	for i in range(6):
		data.append(pd.read_csv('data/s1/original_behaviors_%d.csv'%(i), header=None))

	tmp = data[0].iloc[:,:2].to_numpy(copy=True)
	D = calculate_distances(tmp)

	C, medoids = kmedoids(D, 8)

	a = WTS(data[0].iloc[:,:2].to_numpy(copy=True), C, medoids, D, 0.3)
	new_data = []
	for i in range(1,6):
		new_data.append(data[i].iloc[:,:2].to_numpy(copy=True))

	a.cluster(new_data)

	pickle.dump(a, open("test2.pkl", "wb"))
	a = pickle.load(open("test2.pkl", "rb"))
	#a.plot()

	a.plot_vs_true(data, "test2.png")

def delmh_test_households(z=False, threshold=0.0):
	files = ["1185", "1186", "1198", "1199", "1200", "1201", "1222", "1223", "1224", "1225"]
	best_ks = [5, 4, 3, 8, 4, 4, 4, 3, 3, 4]
	if z==True:
		best_ks = [3, 4, 3, 4, 4, 3, 4, 3, 4, 2]

	# Divide data into segments.
	for file, best_k in zip(files, best_ks):
		d = pd.read_csv('data/delmh/initial_tests/profilers/segmented/%s_0.csv'%(file), index_col=0, parse_dates=True)
		if z == True:
			d=d.apply(zscore, axis=1, result_type='expand')
		D = calculate_distances(d.to_numpy(copy=True))

		if z == True:
			a = pd.read_csv('data/delmh/results/initial_tests/scores/zscore/%s.csv'%(file), index_col=0)
		else:
			a = pd.read_csv('data/delmh/results/initial_tests/scores/%s.csv'%(file), index_col=0)
		C = json.loads(a.loc[best_k]['sil_clusters'])
		medoids = list(set(C))

		c = {}
		for i in range(len(C)):
			if C[i] not in c:
				c[C[i]] = []
			c[C[i]].append(i)
		C = {}
		for i in range(len(medoids)):
			for key in c.keys():
				if key == medoids[i]:
					C[i] = c[key]

		a = WTS(d.to_numpy(copy=True), C, medoids, D, threshold)
		new_data = []
		for i in range(1,6,1):
			j = pd.read_csv('data/delmh/initial_tests/profilers/segmented/%s_%d.csv'%(file, i), index_col=0, parse_dates=True)
			if z == True:
				j = j.apply(zscore, axis=1, result_type='expand')
			new_data.append(j)

		a.cluster(new_data)
		if z == False:
			pickle.dump(a, open('data/delmh/results/initial_tests/%s.pkl'%(file),'wb'))
		else:
			pickle.dump(a, open('data/delmh/results/initial_tests/zscore/%s.pkl'%(file),'wb'))

def delmh_test_households2(z=False, threshold=0.0):
	files = ["KT10", "KT04", "ER01", "GT07", "ER03", "KT06", "KT07", "ER08"]#, "KT05", "ER10"]
	best_ks = [3, 4, 2, 3, 6, 5, 5, 4]
	if z==True:
		best_ks = [4, 3, 3, 3, 5, 5, 4, 3]

	# Divide data into segments.
	for file, best_k in zip(files, best_ks):
		d = pd.read_csv('data/delmh/initial_tests/recorders/segmented/%s_0.csv'%(file), index_col=0, parse_dates=True)
		if z == True:
			d=d.apply(zscore, axis=1, result_type='expand')
		D = calculate_distances(d.to_numpy(copy=True))

		if z == True:
			a = pd.read_csv('data/delmh/results/initial_tests/scores/recorders/zscore/%s.csv'%(file), index_col=0)
		else:
			a = pd.read_csv('data/delmh/results/initial_tests/scores/recorders/%s.csv'%(file), index_col=0)
		C = json.loads(a.loc[best_k]['sil_clusters'])
		medoids = list(set(C))

		c = {}
		for i in range(len(C)):
			if C[i] not in c:
				c[C[i]] = []
			c[C[i]].append(i)

		C = {}
		for i in range(len(medoids)):
			for key in c.keys():
				if key == medoids[i]:
					C[i] = c[key]


		a = WTS(d.to_numpy(copy=True), C, medoids, D, threshold)
		new_data = []
		for i in range(1,6,1):
			j = pd.read_csv('data/delmh/initial_tests/recorders/segmented/%s_%d.csv'%(file, i), index_col=0, parse_dates=True)
			if z == True:
				j = j.apply(zscore, axis=1, result_type='expand')
			new_data.append(j)

		a.cluster(new_data)
		if z == False:
			pickle.dump(a, open('data/delmh/results/initial_tests/recorders/%s.pkl'%(file),'wb'))
		else:
			pickle.dump(a, open('data/delmh/results/initial_tests/recorders/zscore/%s.pkl'%(file),'wb'))


def print_delmh():
	files = ["1185", "1186", "1198", "1199", "1200", "1201", "1222", "1223", "1224", "1225"]
	for file in files:
		a = pickle.load(open('data/delmh/results/initial_tests/%s.pkl'%(file), 'rb'))
		a.plot_delmh('figures/delmh/', file)

	files = ["KT10", "KT04", "ER01", "GT07", "ER03", "KT06", "KT07", "ER08"]#, "KT05", "ER10"]
	for file in files:
		a = pickle.load(open('data/delmh/results/initial_tests/recorders/%s.pkl'%(file), 'rb'))
		a.plot_delmh('figures/delmh/', file)

	files = ["1185", "1186", "1198", "1199", "1200", "1201", "1222", "1223", "1224", "1225"]
	for file in files:
		a = pickle.load(open('data/delmh/results/initial_tests/zscore/%s.pkl'%(file), 'rb'))
		a.plot_delmh('figures/delmh/zscore/', file)

	files = ["KT10", "KT04", "ER01", "GT07", "ER03", "KT06", "KT07", "ER08"]#, "KT05", "ER10"]
	for file in files:
		a = pickle.load(open('data/delmh/results/initial_tests/recorders/zscore/%s.pkl'%(file), 'rb'))
		a.plot_delmh('figures/delmh/zscore/', file)

def s_test(dataset, behavior_type, limit, num_clusters, initial_centroids=None, length=6):
	data = pd.read_csv('data/%s/%s_behaviors_%d.csv'%(dataset, behavior_type, 0), header=None)

	tmp = data.iloc[:,:2].to_numpy(copy=True)
	D = calculate_distances(tmp)

	C, medoids = kmedoids(D, num_clusters, initial_centroids)
	a = WTS(data.iloc[:,:2].to_numpy(copy=True), C, medoids, D, limit)

	new_data = []
	for i in range(1,length):
		new_data.append(pd.read_csv('data/%s/%s_behaviors_%d.csv'%(dataset, behavior_type, i), header=None).iloc[:,:2].to_numpy(copy=True))

	a.cluster(new_data)
	pickle.dump(a, open('data/%s/results/%s.pkl'%(dataset, behavior_type), 'wb'))
	a.to_csv('data/%s/ours/%s.pkl'%(dataset, behavior_type))

def print_s():
	datasets = ['s1', 's2']
	behaviors = ['original','continuous']
	for dataset in datasets:
		for behavior in behaviors:
			data = []
			for i in range(6):
				data.append(pd.read_csv('data/%s/%s_behaviors_%d.csv'%(dataset, behavior, i)))
			a = pickle.load(open("data/%s/results/%s.pkl"%(dataset, behavior), "rb"))
			a.plot_vs_true(data, "figures/%s/%s.png"%(dataset, behavior))


def main():	
	# delmh_test_households()
	# delmh_test_households(True)
	#delmh_test_households2()
	#delmh_test_households2(True)
	#s_test("s1", "original", 0.3, 8, [0, 300, 616, 930, 1248, 1573, 1899, 2233])
	#s_test("s1norm", "original", 0.3, 8, [0, 300, 616, 930, 1248, 1573, 1899, 2233])
	s_test("s11000", "original", 0.3, 4, [0, 300, 616, 930], 5) 
	#s_test("s11000norm", "original", 0.3, 4, [0, 300, 616, 930], 5)

	# s_test("s1", "continuous", 0.3, 15, [0, 150, 308, 465, 624, 786, 949, 1116, 1285, 1455, 1626, 1799, 1973, 2148, 2323])
	# s_test("s2", "original", 0.3, 8, [0, 300, 617, 932, 1252, 1573, 1902, 2236])
	# s_test("s2", "continuous", 0.3, 15, [0, 150, 308, 465, 625, 785, 949, 1116, 1282, 1452, 1624, 1797, 1972, 2147, 2322])
	
	#print_delmh()
	# print_s()

if __name__ == '__main__':
	main()
