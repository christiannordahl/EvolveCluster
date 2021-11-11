import pseudo_random_processes as prp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RBFGenerator():
	def __init__(self, n_clusters=5, n_features=2, n_cluster_range=0, sample_random_state=None, model_random_state=None, kernel_radius=0.07, kernel_radius_range=0.0, drift_speed=0.0, centroid_speeds=None,event_frequency=3000,merge_split=True,create_delete=False, drift_delay=0):
		self.sample_random_state = sample_random_state
		self.model_random_state = model_random_state
		self.n_original_clusters = n_clusters
		self.n_clusters = n_clusters
		self.n_features = n_features
		self.n_current_clusters = n_clusters
		self.n_cluster_range = n_cluster_range
		self.kernel_radius = kernel_radius
		self.kernel_radius_range = kernel_radius_range
		self.drift_speed = drift_speed
		self.centroid_speeds = centroid_speeds
		self.centroids = None
		self.centroid_weights = None
		self.event_frequency=event_frequency
		self.event_calculator=0
		self.merge_split=merge_split
		self.create_delete=create_delete
		self.drift_delay=drift_delay

		self._prepare_for_use()

	def move_centroids(self):
		for i in range(self.n_clusters):
			if self.centroids[i] is not None:
				for j in range(self.n_features):
					self.centroids[i].centre[j] += self.centroids[i].speed[j] * self.drift_speed

					if (self.centroids[i].centre[j] > 1) or (self.centroids[i].centre[j] < 0):
						self.centroids[i].centre[j] = 1 if (self.centroids[i].centre[j] > 1) else 0
						self.centroids[i].speed[j] = -self.centroids[i].speed[j]

	def event(self):

		if self.merge_split and self.create_delete:
			_start = 0
			_stop = 4
		elif self.merge_split:
			_start = 0
			_stop = 2
		elif self.create_delete:
			_start = 2
			_stop = 4
		_start = 2
		_stop = 4
		rand  = self._sample_random_state.randint(_start, _stop)

		if rand == 0:	# Merge
			self.merge()
		elif rand == 1:	# Split
			self.split()
		elif rand == 2:	# Create
			if self.n_current_clusters < (self.n_original_clusters + self.n_cluster_range):
				self.create()
			else:
				self.delete()
		elif rand == 3:	# Delete
			if self.n_current_clusters > (self.n_original_clusters - self.n_cluster_range):
				self.delete()
			else:
				self.create()

	def merge(self):
		pass
	def split(self):
		pass
	def create(self):
		self.centroids.append(Centroid())
		rand_centre = []
		for j in range(self.n_features):
			rand_centre.append(self._sample_random_state.rand())
		self.centroids[-1].centre = rand_centre
		self.centroids[-1].class_label = len(self.centroids)-1
		#self.centroids[i].std_dev = model_random_state.rand()
		# Calculates a varying kernel radius, based on the kernel radius range variable.
		# If the radius range is 0, then std_dev is always same as kernel radius.
		self.centroids[-1].std_dev = self.kernel_radius + (self._sample_random_state.randint(3)-1)*self.kernel_radius_range
		self.centroid_weights.append(self._sample_random_state.rand())
		self.n_current_clusters += 1
		self.n_clusters += 1

		if self.centroid_speeds is None:
			rand_speed = []
			norm_speed = 0.0
			for j in range(self.n_features):
				rand_speed.append(self._sample_random_state.rand())
				norm_speed += rand_speed[j]*rand_speed[j]
			norm_speed = np.sqrt(norm_speed)
			for j in range(self.n_features):
				rand_speed[j] /= norm_speed
			self.centroids[-1].speed = rand_speed
		else:
			self.centroids[-1].speed = self.centroid_speeds

	def delete(self):
		i = prp.random_index_based_on_weights(self.centroid_weights, self._sample_random_state)
		while self.centroids[i] is None:
			i = prp.random_index_based_on_weights(self.centroid_weights, self._sample_random_state)

		self.n_current_clusters -= 1
		self.centroids[i] = None
		self.centroid_weights[i] = 0.0


	def next_sample(self, batch_size=1):
		data = np.zeros([batch_size, self.n_features+1])
		if self.drift_delay > 0:
			self.drift_delay -= 1
		elif self.event_calculator == self.event_frequency:
			self.event()
			self.event_calculator = 0

		for j in range(batch_size):
			self.move_centroids()
			i = prp.random_index_based_on_weights(self.centroid_weights, self._sample_random_state)
			while self.centroids[i] is None:
				i = prp.random_index_based_on_weights(self.centroid_weights, self._sample_random_state)

			centroid_aux = self.centroids[i]
			att_vals = []
			magnitude = 0.0
			for i in range(self.n_features):
				att_vals.append((self._sample_random_state.rand()*2.0)-1.0)
				magnitude += att_vals[i] * att_vals[i]
			magnitude = np.sqrt(magnitude)
			desired_mag = self._sample_random_state.normal() * centroid_aux.std_dev
			scale = desired_mag/magnitude
			for i in range(self.n_features):
				data[j, i] = centroid_aux.centre[i] + att_vals[i] * scale
			data[j, self.n_features] = centroid_aux.class_label
			self.event_calculator += 1
		self.current_sample_x = data[:, :self.n_features]
		self.current_sample_y = data[:, self.n_features:].flatten().astype(int)

		#return self.current_sample_x, self.current_sample_y
		data = data.flatten()
		return data

	def _prepare_for_use(self):
		self._generate_centroids()
		self._sample_random_state = prp.check_random_state(self.sample_random_state)

	def _generate_centroids(self):
		model_random_state = prp.check_random_state(self.model_random_state)
		self.centroids = []
		self.centroid_weights = []
		for i in range(self.n_clusters):
			self.centroids.append(Centroid())
			rand_centre = []
			for j in range(self.n_features):
				rand_centre.append(model_random_state.rand())
			self.centroids[i].centre = rand_centre
			self.centroids[i].class_label = i
			#self.centroids[i].std_dev = model_random_state.rand()
			# Calculates a varying kernel radius, based on the kernel radius range variable.
			# If the radius range is 0, then std_dev is always same as kernel radius.
			self.centroids[i].std_dev = self.kernel_radius + (model_random_state.randint(3)-1)*self.kernel_radius_range
			self.centroid_weights.append(model_random_state.rand())

		for i in range(self.n_clusters):
			# Constant drift of centroids
			if self.centroid_speeds is None:
				rand_speed = []
				norm_speed = 0.0
				for j in range(self.n_features):
					rand_speed.append(model_random_state.rand())
					norm_speed += rand_speed[j]*rand_speed[j]
				norm_speed = np.sqrt(norm_speed)
				for j in range(self.n_features):
					rand_speed[j] /= norm_speed
				self.centroids[i].speed = rand_speed
			else:
				self.centroids[i].speed = self.centroid_speeds


class Centroid():
	def __init__(self):
		self.centre = None
		self.std_dev = None
		self.class_label = None
		self.speed = None

def generate_data(number_of_instances=5000, n_clusters=5, n_features=2, n_cluster_range=3, sample_random_state=99, model_random_state=50, kernel_radius=0.02, kernel_radius_range=0.005,drift_speed=0.0001,centroid_speeds=None,event_frequency=2500,merge_split=False,create_delete=True, drift_delay=0):
	r = RBFGenerator(n_clusters, 
					n_features, 
					n_cluster_range, 
					sample_random_state, 
					model_random_state, 
					kernel_radius, 
					kernel_radius_range,
					drift_speed,
					centroid_speeds,
					event_frequency,
					merge_split,
					create_delete,
					drift_delay)
	a = []
	for i in range(number_of_instances):
		a.append(r.next_sample())

	a = pd.DataFrame(a)
	a.columns = [*a.columns[:-1], 'cluster']
	a.cluster = a.cluster.astype(int)
	return a

def main():
	seeds = [1,6,2,12,29]
	for seed in seeds:
		print(seed)
		number_of_instances = 10000
		data = generate_data(number_of_instances,
						n_clusters=5, 
						n_features=2, 
						n_cluster_range=3, 
						sample_random_state=seed, 
						model_random_state=10, 
						kernel_radius=0.02, 
						kernel_radius_range=0.005,
						drift_speed=0.0001,
						centroid_speeds=None,
						event_frequency=2000,
						merge_split=False,
						create_delete=True,
						drift_delay=500)
		
		for i in range(5):
			dat = data.iloc[(number_of_instances//5)*i:(number_of_instances//5)*(i+1),:]
			dat.to_csv('%d_2-dim_%d.csv'%(seed,i), index=False)

	return
	number_of_instances = 10000
	data = generate_data(number_of_instances,
					n_clusters=5, 
					n_features=8, 
					n_cluster_range=4, 
					sample_random_state=99, 
					model_random_state=50, 
					kernel_radius=0.02, 
					kernel_radius_range=0.005,
					drift_speed=0.0001,
					centroid_speeds=None,
					event_frequency=2000,
					merge_split=False,
					create_delete=True)
	
	#a = data
	#data = data.iloc[4000:6000,:]
	#data.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')
	#plt.xlim(-0.2,1.2)
	#plt.ylim(-0.2,1.2)
	#plt.show()
	for i in range(5):
		dat = data.iloc[(number_of_instances//5)*i:(number_of_instances//5)*(i+1),:]
		dat.to_csv('8-dim_%d.csv'%(i), index=False)

	number_of_instances = 10000
	data = generate_data(number_of_instances,
					n_clusters=5, 
					n_features=14, 
					n_cluster_range=4, 
					sample_random_state=99, 
					model_random_state=50, 
					kernel_radius=0.02, 
					kernel_radius_range=0.005,
					drift_speed=0.0001,
					centroid_speeds=None,
					event_frequency=2000,
					merge_split=False,
					create_delete=True)
	
	#a = data
	#data = data.iloc[4000:6000,:]
	#data.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')
	#plt.xlim(-0.2,1.2)
	#plt.ylim(-0.2,1.2)
	#plt.show()
	for i in range(5):
		dat = data.iloc[(number_of_instances//5)*i:(number_of_instances//5)*(i+1),:]
		dat.to_csv('14-dim_%d.csv'%(i), index=False)

if __name__ == '__main__':
	main()
