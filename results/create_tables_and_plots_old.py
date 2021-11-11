import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import jaccard_similarity_score
import pickle

def scatterplots(data, limits, minmax, dataset, behavior):
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
		plt.xlim(minmax[0])
		plt.ylim(minmax[1])
		plt.savefig('../figures/%s/%s_%d.pdf'%(dataset,behavior,i))
		plt.close()

def new_scatterplots(data, limits, minmax, dataset, behavior):
	for i in range(len(data)):
		plt.scatter(data[i]['x'], 
					data[i]['y'], 
					c=data[i]['cluster'], 
					cmap=plt.cm.tab20b,
					s=10,
					marker='o')
		#plt.xlim(minmax[0])
		#plt.ylim(minmax[1])
		plt.savefig('../figures/%s/%s_%d.pdf'%(dataset,behavior,i))
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
	datasets = ['S1','S2']
	date = '201210'
	behaviors = ['original','continuous']
	minmax = [[[15000,980000],[45000,980000]],[[50000,990000],[20000,995000]]]
	# x min/max s1, ymin/max s1, xmin/max s2, ymin/max s2

	j=0
	for dataset in datasets:
		for behavior in behaviors:
			data = [pd.read_csv('%s/%s/%s_results_%d.csv'%(dataset, date, behavior, i), index_col=False) for i in range(6)]
			kmeans = [pickle.load(open('%s/%s/%s_results_%d.pkl'%(dataset, date, behavior, i),'rb')) for i in range(6)]
			limits = [[len(data[i-1]) if i > 0 else 0,len(data[i])] for i in range(6)]


			new_scatterplots(data, limits, minmax[j], dataset, behavior)
			#tables(data, dataset, behavior)
			
		j+=1

if __name__ == '__main__':
	main()
