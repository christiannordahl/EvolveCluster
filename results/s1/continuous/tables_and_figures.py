import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import json
import matplotlib.pyplot as plt
import sys

def figures_and_tables_single():
	pivot_F,pivot_J,pivot_S = [],[],[]
	pivot_files = ['PivotBi/PivotBiCluster-result%d.json'%(x) for x in range(4)]
	pivot_filter_files = ['PivotBi/firstSet%d.json'%(x) for x in range(4)]
	pivot_plot_files = [pd.read_json('PivotBi/firstSet%d.json'%(x)) for x in range(5)]
	for i in range(4):
		df = pd.read_json(pivot_files[i])

	split_F,split_J,split_S = [],[],[]
	split_files = ['SplitMerge/Split-Merge-result%d.json'%(x) for x in range(4)]
	split_filter_files = ['SplitMerge/firstSet%d.json'%(x) for x in range(4)]
	split_plot_files = [pd.read_json('SplitMerge/firstSet%d.json'%(x)) for x in range(5)]
	for i in range(4):
		df = pd.read_json(split_files[i])

	evolve_F, evolve_J, evolve_S = [],[],[]
	evolve_files = []
	evolve_filter_files = []
	evolve_plot_files = [pd.read_csv('%d.csv'%(x), index_col=0) for x in range(5)]

	plot_files = [evolve_plot_files,split_plot_files,pivot_plot_files]
	top_labels = ["D$_0$","D$_1$","D$_2$","D$_3$","D$_4$"]
	plot_combined(plot_files, 'continuous_single.pdf', top_labels)

	basefile = pd.read_csv('basefile_0.csv')
	for i in range(4):
		a = pd.read_csv('basefile_%d.csv'%(i+1))
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)

	for i in range(5):
		df = pivot_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		pivot_F.append(cluster_wise_f_measure(df))
		pivot_J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			pivot_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			pivot_S.append(-11)
	data = [pivot_F, pivot_J, pivot_S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=top_labels)
	table_df.to_latex(open("PivotBi_single.tex", "w"), float_format="%.5f", caption="asd", label="tab:s1-cont-pivotbi-single", escape=False)

	for i in range(5):
		df = split_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		split_F.append(cluster_wise_f_measure(df))
		split_J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			split_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			split_S.append(-11)
	data = [split_F, split_J, split_S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=top_labels)
	table_df.to_latex(open("SplitMerge_single.tex", "w"), float_format="%.5f", caption="asd", label="tab:s1-cont-splitmerge-single", escape=False)

	for i in range(5):
		df = evolve_plot_files[i]
		df['cluster'] = df['cluster_ids']
		#df = df.merge(basefile, on='_id', how='inner')
		evolve_F.append(cluster_wise_f_measure(df))
		evolve_J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			evolve_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			evolve_S.append(-11)
	data = [evolve_F, evolve_J, evolve_S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=top_labels)
	table_df.to_latex(open("EvolveCluster_single.tex", "w"), float_format="%.5f", caption="asd", label="tab:s1-cont-eca-single", escape=False)

def figures_and_tables_combined():
	pivot_F,pivot_J,pivot_S = [],[],[]
	pivot_plot_files = [pd.read_json('PivotBi/PivotBiCluster-result%d.json'%(x)) for x in range(4)]

	split_F,split_J,split_S = [],[],[]
	split_plot_files = [pd.read_json('SplitMerge/Split-Merge-result%d.json'%(x)) for x in range(4)]

	evolve_F, evolve_J, evolve_S = [],[],[]
	evolve_plot_files = [pd.read_csv('%d.csv'%(x), index_col=0) for x in range(5)]
	for i in range(4):
		evolve_plot_files[i] = evolve_plot_files[i].append(evolve_plot_files[i+1], ignore_index=True)
	del[evolve_plot_files[4]]

	plot_files = [evolve_plot_files,split_plot_files,pivot_plot_files]
	top_labels = ["D$_0$ - D$_1$","D$_1$ - D$_2$","D$_2$ - D$_3$","D$_3$ - D$_4$"]
	plot_combined(plot_files, 'continuous_combined.pdf', top_labels)
	
	basefile = pd.read_csv('basefile_0.csv')
	for i in range(4):
		a = pd.read_csv('basefile_%d.csv'%(i+1))
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)

	for i in range(4):
		df = pivot_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		pivot_F.append(cluster_wise_f_measure(df))
		pivot_J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			pivot_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			pivot_S.append(-11)
	data = [pivot_F, pivot_J, pivot_S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=top_labels)
	table_df.to_latex(open("PivotBi_combined.tex", "w"), float_format="%.5f", caption="asd", label="tab:s1-cont-pivotbi-combined", escape=False)
	for i in range(4):
		df = split_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		split_F.append(cluster_wise_f_measure(df))
		split_J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			split_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			split_S.append(-11)
	data = [split_F, split_J, split_S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=top_labels)
	table_df.to_latex(open("SplitMerge_combined.tex", "w"), float_format="%.5f", caption="asd", label="tab:s1-cont-splitmerge-combined", escape=False)
	for i in range(4):
		df = evolve_plot_files[i]
		df['cluster'] = df['cluster_ids']
		evolve_F.append(cluster_wise_f_measure(df))
		evolve_J.append(cluster_wise_jaccard(df))
		D = calculate_distances(df[['x','y']].to_numpy(copy=True))
		try:
			evolve_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			evolve_S.append(-11)
	data = [evolve_F, evolve_J, evolve_S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=top_labels)
	table_df.to_latex(open("EvolveCluster_combined.tex", "w"), float_format="%.5f", caption="asd", label="tab:s1-cont-eca-combined", escape=False)

def plot_combined(data, output, top_labels):
	char = 97
	figsize = (12,6)
	if(len(data[0]) > 4):
		figsize = (15,6)
	fig, axs = plt.subplots(len(data),len(data[0]),figsize=figsize, sharex=True, sharey=True)
	for i in range(len(data)):
		axs[i][0].set_ylabel(chr(char)+')', rotation=0, fontsize=20, labelpad=20, va='center')
		char += 1
		for j in range(len(data[i])):
			if i == 0:
				axs[i][j].set_title(top_labels[j], fontsize=20)
			if 'attributes' in data[i][j].columns:
				axs[i][j].scatter(data[i][j]['attributes'].map(lambda x: x[0]), 
						data[i][j]['attributes'].map(lambda x: x[1]), 
						c=data[i][j]['cluster'], cmap='tab20', s=4, alpha=0.5)
			else:
				axs[i][j].scatter(data[i][j]['x'], 
						data[i][j]['y'], 
						c=data[i][j]['cluster_ids'], cmap='tab20', s=4,alpha=0.5)

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
	figures_and_tables_combined()
	figures_and_tables_single()

if __name__ == '__main__':
	main()
