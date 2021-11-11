import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import jaccard_similarity_score
import pickle

from sklearn.cluster import MiniBatchKMeans

def scatterplots(data, limits, minmax, dataset, behavior, kmeans):
	for i in range(len(data)):
		plt.scatter(data[i].loc[:limits[i][0],'x'], 
					data[i].loc[:limits[i][0],'y'], 
					c=data[i].loc[:limits[i][0],'cluster'], 
					cmap=plt.cm.tab20b,
					s=10,
					marker='_',
					alpha=0.5)
		plt.scatter(data[i].loc[limits[i][0]:limits[i][1],'x'], 
					data[i].loc[limits[i][0]:limits[i][1],'y'], 
					c=data[i].loc[limits[i][0]:limits[i][1],'cluster'], 
					cmap=plt.cm.tab20b,
					s=10,
					marker='o')
		centers = kmeans[i].cluster_centers_
		x,y = [],[]
		for center in centers:
			x.append(center[0])
			y.append(center[1])
		plt.scatter(x,y, color='black')
		plt.xlim(minmax[0])
		plt.ylim(minmax[1])
		plt.savefig('../figures/%s/%s_%d.pdf'%(dataset,behavior,i))
		plt.close()
		plt.scatter(data[i].loc[limits[i][0]:limits[i][1],'x'], 
					data[i].loc[limits[i][0]:limits[i][1],'y'], 
					c=data[i].loc[limits[i][0]:limits[i][1],'cluster'], 
					cmap=plt.cm.tab20b,
					s=10,
					marker='o')
		plt.scatter(x,y, color='black')
		plt.xlim(minmax[0])
		plt.ylim(minmax[1])
		plt.savefig('../figures/%s/%s_%d_separate.pdf'%(dataset,behavior,i))
		plt.close()

		plt.scatter(data[i]['x'], 
					data[i]['y'], 
					c=data[i]['cluster'], 
					cmap=plt.cm.tab20b,
					s=10,
					marker='_',
					alpha=0.5)
		for center in centers:
			x.append(center[0])
			y.append(center[1])
		plt.scatter(x,y, color='black')
		plt.xlim(minmax[0])
		plt.ylim(minmax[1])
		plt.savefig('../figures/%s/%s_%d_combined.pdf'%(dataset,behavior,i))
		plt.close()


def new_scatterplots(data, limits, minmax, dataset, behavior, kmeans):
	for i in range(len(data)):
		print(data[i])
		plt.scatter(data[i].iloc[:,0], 
					data[i].iloc[:,1], 
					c=data[i]['cluster'], 
					cmap=plt.cm.tab20b,
					s=10,
					marker='o')
		plt.xlim(minmax[0])
		plt.ylim(minmax[1])
		centers = data[i]['cluster'].unique()
		x,y = [],[]
		for center in centers:
			x.append(data[i].iloc[center,:][0])
			y.append(data[i].iloc[center,:][1])
		#plt.scatter(x,y, color='black')
		plt.savefig('../figures/%d.pdf'%(i))
		plt.close()



def tables(data, dataset, behavior):
	data[0]['cluster'] = data[0]['cluster']+1

	cols = ['Chunk %d'%(i) for i in range(len(data))]
	df = pd.DataFrame(columns=cols)
	scores = []
	for i in range(len(data)):
		scores.append(adjusted_rand_score(data[i]['actual_cluster'],data[i]['cluster']))
	df.loc[0] = scores
	scores = []
	for i in range(len(data)):
		scores.append(jaccard_similarity_score(data[i]['actual_cluster'],data[i]['cluster']))
	df.loc[1] = scores
	scores = []
	for i in range(len(data)):
		scores.append(silhouette_score(data[i].iloc[:,:2],data[i]['cluster']))
	df.loc[2] = scores

	with open('../tables/%s/%s.tex'%(dataset, behavior), 'w') as f:
		f.write(df.to_latex())


def main():

	data = [pd.read_csv('../data/s1_50_50/ours/res%d.csv'%(x), index_col=False) for x in range(2)]
	new_scatterplots(data, None, [[-0.1,1.1],[-0.1,1.1]], None, None, None)

	return

	datasets = ['s1','s11000', 's11000hakan', 's11000hakannorm', 's11000norm', 's1hakan', 's1norm']
	new_func = [False, False, True, True, False, True, False]	
	lengths = [6, 5, 4, 3, 5, 2, 6]
	date = '201211'
	behaviors = ['original']
	minmax = [[[15000,980000],[45000,980000]],
				[[15000,980000],[45000,980000]],
				[[15000,980000],[45000,980000]],
				[[-0.1,1.1],[-0.1,1.1]],
				[[-0.1,1.1],[-0.1,1.1]],
				[[15000,980000],[45000,980000]],
				[[-0.1,1.1],[-0.1,1.1]]]
	# x min/max s1, ymin/max s1, 

	#[[50000,990000],[20000,995000]]]
	#xmin/max s2, ymin/max s2
	j=0
	for dataset in datasets:
		for behavior in behaviors:
			data = [pd.read_csv('../data/%s/%s_results_%d.csv'%(dataset, behavior, i), index_col=False) for i in range(lengths[j])]
			kmeans = [pickle.load(open('../data/%s/%s_results_%d.pkl'%( dataset, behavior, i),'rb')) for i in range(lengths[j])]
			limits = [[len(data[i-1]) if i > 0 else 0,len(data[i])] for i in range(lengths[j])]

			if new_func[j] is True:
				new_scatterplots(data, limits, minmax[j], dataset, behavior, kmeans)
			else:
				scatterplots(data, limits, minmax[j], dataset, behavior, kmeans)
			#tables(data, dataset, behavior)
			
			j+=1

if __name__ == '__main__':
	main()
