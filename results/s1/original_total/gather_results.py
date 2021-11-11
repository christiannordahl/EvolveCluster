import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import json
import matplotlib.pyplot as plt

def PivotBi():
	NUM_FILES = 4
	F,J,S = [],[],[]
	files = ['PivotBiCluster-result%d.json'%(x) for x in range(4)]
	filter_files = ['PivotBi/firstSet%d.json'%(x) for x in range(4)]
	plot_files = [pd.read_json('PivotBi/firstSet%d.json'%(x)) for x in range(5)]
	
	for i in range(4):
		df = pd.read_json(files[i])
		plot_files[i+1] = df[~df['_id'].isin(plot_files[i+1]['_id'])]	

	sessions = json.load(open('PivotBi/session_terms.json','r'))
	for i in range(len(sessions)-1,-1,-1):
		if 'firstSet' not in list(sessions[i].keys())[0]:
			del sessions[i]

	clusters = []
	for i in range(len(sessions)):
		centroids = []
		for j in range(len(sessions[i]['firstSet%d'%(i)].keys())):
			centroids.append(sessions[i]['firstSet%d'%(i)][str(j)])
		clusters.append(centroids)

	plot(plot_files, clusters, 'PivotBi.pdf')

	basefile = pd.read_csv('../0.csv')
	for i in range(NUM_FILES):
		a = pd.read_csv('../%d.csv'%(i+1))
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)

	i = 0
	for i in range(5):
		df = plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')

		F.append(cluster_wise_f_measure(df))
		J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			S.append(-11)
		

	data = [F, J, S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=["D$_0$","D$_1$","D$_2$","D$_3$","D$_4$"])
	table_df.to_latex(open("PivotBi.tex", "w"), float_format="%.5f", caption="asd", label="asd")

def SplitMerge():
	NUM_FILES = 4
	F,J,S = [],[],[]
	files = ['Split-Merge-result%d.json'%(x) for x in range(4)]
	filter_files = ['SplitMerge/firstSet%d.json'%(x) for x in range(4)]
	plot_files = [pd.read_json('SplitMerge/firstSet%d.json'%(x)) for x in range(5)]

	for i in range(4):
		df = pd.read_json(files[i])
		plot_files[i+1] = df[~df['_id'].isin(plot_files[i+1]['_id'])]

	sessions = json.load(open('SplitMerge/session_terms.json','r'))
	for i in range(len(sessions)-1,-1,-1):
		if 'firstSet' not in list(sessions[i].keys())[0]:
			del sessions[i]

	clusters = []
	for i in range(len(sessions)):
		centroids = []
		for j in range(len(sessions[i]['firstSet%d'%(i)].keys())):
			centroids.append(sessions[i]['firstSet%d'%(i)][str(j)])
		clusters.append(centroids)

	plot(plot_files, clusters, 'SplitMerge.pdf')

	basefile = pd.read_csv('../0.csv')
	for i in range(NUM_FILES):
		a = pd.read_csv('../%d.csv'%(i+1))
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)

	i = 0
	# for file, filter_file in zip(files, filter_files):
	# 	df = pd.read_json(file)
	# 	filter_df = pd.read_json(filter_file)
	# 	df.drop(columns=['attributes'],inplace=True)
	# 	if(df['cluster'].min() != 0):
	# 		df['cluster'] -= 1

	# 	df = df[~df['_id'].isin(filter_df['_id'])]
	# 	df = df.merge(basefile, on='_id', how='inner')
	for i in range(5):
		df = plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')

		F.append(cluster_wise_f_measure(df))
		J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			S.append(-11)
		

	data = [F, J, S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=["D$_0$","D$_1$","D$_2$","D$_3$","D$_4$"])
	table_df.to_latex(open("SplitMerge.tex", "w"), float_format="%.5f", caption="", label="")

def plot(data, clusters, output):
	colors=['teal', 'aqua', 'pink', 'brown', 'fuchsia', 'gold', 'lime', 'orange', 'purple', 'red', 'yellow', 'khaki', 'grey', 'tan', 'coral']
	fig, axs = plt.subplots(1,len(data),figsize=(20,3), sharex=True, sharey=True)
	for i in range(len(data)):
		axs[i].scatter(data[i]['attributes'].map(lambda x: x[0]), 
						data[i]['attributes'].map(lambda x: x[1]), 
						c=data[i]['cluster'], cmap='tab20')
		# for j in range(len(clusters[i])):
		# 	axs[i].scatter(clusters[i][j][0],clusters[i][j][1],color='black')
		#for cluster in clusters::
		#	for index in self.clusters[i][key]:
		#		axs[i].scatter(self.data[i][index][0], self.data[i][index][1], color=colors[key], s=30)
		#	axs[i].scatter(self.data[i][self.centroids[i][key]][0], self.data[i][self.centroids[i][key]][1], color=colors[key%len(colors)],edgecolors='black', linewidth=1, s=30)

		plt.ylim(-0.1,1.1)
		plt.xlim(-0.1,1.1)
	plt.tight_layout()
	fig.savefig(output, bbox_inches='tight')
	plt.close()

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
    # unique_clusters_predicted = sorted(predicted["cluster"].unique())

    # for cluster in unique_clusters_predicted:
    #     predicted_cluster = predicted[predicted["cluster"] == cluster]
    #     true_cluster = cluster_with_max_shared_experts(predicted_cluster, true)
    #     sum = sum + f_measure(set(predicted_cluster["_id"]), set(true_cluster["_id"]))
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
	
	# predicted_experts = set(predicted_cluster["_id"])
	# true_sessions = sorted(experts["cluster"].unique())
	# max_cluster = 0
	# max_session = 0
	# for disease in true_sessions:
	# 	cluster = experts[experts["cluster"] == disease]
	# 	same_experts = len(predicted_experts & set(cluster["_id"]))
	# 	if same_experts > max_cluster:
	# 		max_cluster = same_experts
	# 		max_session = disease

	# return experts[experts["cluster"] == max_session]

def s_new_indiv_metrics(dataset, length):
	data = [pd.read_csv('data/s1new/%s/results/%d.csv'%(dataset, i), index_col=0) for i in range(length)]
	F = []
	J = []
	S = []
	for i in range(len(data)):
		f = cluster_wise_f_measure(data[i])
		F.append(f)

		j = cluster_wise_jaccard(data[i])
		J.append(j)

		D = calculate_distances(data[i].iloc[:,:2].to_numpy(copy=True))
		S.append(silhouette_score(D, data[i]["cluster"], metric="precomputed"))
	
	data = [F, J, S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=["D$_0$","D$_1$","D$_2$","D$_3$","D$_4$"])
	table_df.to_latex(open("tables/s1/%s.tex"%(dataset), "w"), float_format="%.5f", caption="", label="")


def main():
	PivotBi()
	SplitMerge()

if __name__ == '__main__':
	main()
