import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import json
import matplotlib.pyplot as plt

def PivotBi():
	NUM_FILES = 1
	F,J,S = [],[],[]
	num_clusters = []
	files = ['PivotBi/PivotBiCluster-result%d.json'%(x) for x in range(1)]
	filter_files = ['PivotBi/firstSet%d.json'%(x) for x in range(1)]
	plot_files = [pd.read_json('PivotBi/firstSet%d.json'%(x)) for x in range(1)]
	for i in range(1):
		df = pd.read_json(files[i])
		plot_files.append(df[~df['_id'].isin(plot_files[i]['_id'])])

	basefile = pd.read_csv('basefile_0.csv', index_col=0)
	for i in range(NUM_FILES):
		a = pd.read_csv('basefile_%d.csv'%(i+1), index_col=0)
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)

	plot_files.append(pd.read_json('PivotBi/PivotBiCluster-result0.json'))
	for i in range(3):
		df = plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		num_clusters.append([len(df.cluster.unique()),df.cluster.unique()])
		F.append(cluster_wise_f_measure(df))
		J.append(cluster_wise_jaccard(df))

		D = calculate_distances(df.iloc[:,3:17].to_numpy(copy=True))
		try:
			S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			S.append(-11)

	data = [F, J, S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=["D$_0$","D$_1$", "D$_0$ - D$_1$"])
	table_df.to_latex(open("PivotBi.tex", "w"), float_format="%.5f", caption="", label="")

def SplitMerge():
	NUM_FILES = 1
	F,J,S = [],[],[]
	num_clusters = []
	files = ['SplitMerge/Split-Merge-result%d.json'%(x) for x in range(1)]
	filter_files = ['SplitMerge/firstSet%d.json'%(x) for x in range(1)]
	plot_files = [pd.read_json('SplitMerge/firstSet%d.json'%(x)) for x in range(1)]
	for i in range(1):
		df = pd.read_json(files[i])
		plot_files.append(df[~df['_id'].isin(plot_files[i]['_id'])])

	basefile = pd.read_csv('basefile_0.csv', index_col=0)
	for i in range(NUM_FILES):
		a = pd.read_csv('basefile_%d.csv'%(i+1), index_col=0)
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)


	plot_files.append(pd.read_json('SplitMerge/Split-Merge-result0.json'))
	for i in range(3):
		df = plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		num_clusters.append([len(df.cluster.unique()),df.cluster.unique()])
		F.append(cluster_wise_f_measure(df))
		J.append(cluster_wise_jaccard(df))

		D = calculate_distances(df.iloc[:,3:17].to_numpy(copy=True))
		try:
			S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			S.append(-11)

	data = [F, J, S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=["D$_0$","D$_1$", "D$_0$ - D$_1$"])
	table_df.to_latex(open("SplitMerge.tex", "w"), float_format="%.5f", caption="", label="")

def ECA():
	F,J,S = [],[],[]
	num_clusters = []
	files = ['0.csv','1.csv']
	for i in range(len(files)):
		df = pd.read_csv(files[i], index_col=0)
		df = df.drop(df.columns[[0]], axis=1)
		df['cluster'] = df['cluster_ids']
		F.append(cluster_wise_f_measure(df))
		J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df.iloc[:,:14])
		try:
			S.append(silhouette_score(D, df['cluster'], metric='precomputed'))
		except ValueError:
			S.append(-11)
		num_clusters.append([len(df.cluster.unique()),df.cluster.unique()])
	df = pd.read_csv(files[1], index_col=0)
	df = df.drop(df.columns[[0]], axis=1)
	id_fixer = [(0,3),(1,5),(2,6)]
	temp = pd.read_csv(files[0], index_col=0)
	temp = temp.drop(temp.columns[[0]], axis=1)
	for j in range(len(id_fixer)):
		df.loc[df['cluster_ids'] == id_fixer[j][0], 'cluster_ids'] = id_fixer[j][1]
	df = temp.append(df, ignore_index=True)
	df['cluster'] = df['cluster_ids']
	num_clusters.append([len(df.cluster.unique()),df.cluster.unique()])

	F.append(cluster_wise_f_measure(df))
	J.append(cluster_wise_jaccard(df))
	D = calculate_distances(df.iloc[:,:14])
	try:
		S.append(silhouette_score(D, df['cluster'], metric='precomputed'))
	except ValueError:
		S.append(-11)

	data = [F, J, S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=["D$_0$","D$_1$", "D$_0$ - D$_1$"])
	table_df.to_latex(open("ECA.tex", "w"), float_format="%.5f", caption="", label="")

def calculate_distances(data):
    return pairwise_distances(data,data)

def cluster_wise_jaccard(data):
	sum = 0
	unique_clusters_predicted = sorted(data['cluster'].unique())
	for cluster in unique_clusters_predicted:
		predicted_cluster = data[data['cluster'] == cluster]
		true_cluster = cluster_with_max_shared(predicted_cluster, data)
		sum += jaccard_measure(set(predicted_cluster['_id']), set(true_cluster['_id']))

	length = len(unique_clusters_predicted)
	return (sum / length)

def jaccard_measure(pred, true):
	value = (len(pred & true)) / (len(true) + len(pred) - len(true & pred))
	return value

def cluster_wise_f_measure(data):
    sum = 0
    unique_clusters_predicted = sorted(data['cluster'].unique())
    for cluster in unique_clusters_predicted:
    	predicted_cluster = data[data['cluster'] == cluster]
    	true_cluster = cluster_with_max_shared(predicted_cluster, data)
    	sum += f_measure(set(predicted_cluster['_id']), set(true_cluster['_id']))

    length = len(unique_clusters_predicted)
    return (sum / length)

def f_measure(pred, true):
    value = (2 * len(pred & true)) / (len(true) + len(pred))
    return value

def cluster_with_max_shared(predicted_cluster, data):
	predicted = set(predicted_cluster['_id'])
	true = data['actual_cluster'].unique()
	max_cluster = 0
	max_session = 0

	for label in true:
		cluster = data[data['actual_cluster'] == label]
		same_cluster = len(predicted & set(cluster['_id']))
		if same_cluster > max_cluster:
			max_cluster = same_cluster
			max_session = label
	return data[data['actual_cluster'] == max_session]

def main():
	PivotBi()
	SplitMerge()
	ECA()

if __name__ == '__main__':
	main()
