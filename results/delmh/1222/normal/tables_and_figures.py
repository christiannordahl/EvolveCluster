import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import json
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import numpy as np
import copy

def figures_and_tables_single():
	pivot_I,pivot_S = [],[]
	pivot_files = ['PivotBi/PivotBiCluster-result%d.json'%(x) for x in range(4)]
	pivot_filter_files = ['PivotBi/firstSet%d.json'%(x) for x in range(4)]
	pivot_plot_files = [pd.read_json('PivotBi/firstSet%d.json'%(x)) for x in range(5)]

	split_I,split_S = [],[]
	split_files = ['SplitMerge/Split-Merge-result%d.json'%(x) for x in range(4)]
	split_filter_files = ['SplitMerge/firstSet%d.json'%(x) for x in range(4)]
	split_plot_files = [pd.read_json('SplitMerge/firstSet%d.json'%(x)) for x in range(5)]

	evolve_I,evolve_S = [],[]
	evolve_files = []
	evolve_filter_files = []
	evolve_plot_files = [pd.read_csv('EvolveCluster/%d.csv'%(x)) for x in range(5)]

	plot_files = [evolve_plot_files,split_plot_files,pivot_plot_files]
	top_labels = ["D$_0$","D$_1$","D$_2$","D$_3$","D$_4$"]

	basefile = pd.read_csv('0.csv', index_col=0, parse_dates=True)
	for i in range(4):
		a = pd.read_csv('%d.csv'%(i+1), index_col=0, parse_dates=True)
		basefile = basefile.append(a, ignore_index=True)

	# if(basefile['cluster'].min() != 0):
	# 	basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)

	columns=[str(x) for x in range(24)]
	for i in range(5):
		df = pivot_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		D = calculate_distances(df[columns])
		pivot_I.append(IC_av(pd.DataFrame(D), df["cluster"])[0])
		try:
			pivot_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			pivot_S.append(-11)
	data = [pivot_S, pivot_I]
	table_df = pd.DataFrame(data, index=["Silhouette", "IC-av"], columns=top_labels)
	table_df.to_latex(open("PivotBi_single.tex", "w"), float_format="%.5f", caption="asd", label="tab:delmh-orig-pivotbi-single", escape=False)

	for i in range(5):
		df = split_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		D = calculate_distances(df[columns])
		split_I.append(IC_av(pd.DataFrame(D), df["cluster"])[0])
		try:
			split_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			split_S.append(-11)
	data = [split_S, split_I]
	table_df = pd.DataFrame(data, index=["Silhouette", "IC-av"], columns=top_labels)
	table_df.to_latex(open("SplitMerge_single.tex", "w"), float_format="%.5f", caption="asd", label="tab:delmh-orig-splitmerge-single", escape=False)

	for i in range(5):
		df = evolve_plot_files[i]
		df['cluster'] = df['cluster_ids']
		D = calculate_distances(df[columns])
		evolve_I.append(IC_av(pd.DataFrame(D), df["cluster"])[0])
		try:
			evolve_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			evolve_S.append(-11)
	data = [evolve_S, evolve_I]
	table_df = pd.DataFrame(data, index=["Silhouette", "IC-av"], columns=top_labels)
	table_df.to_latex(open("EvolveCluster_single.tex", "w"), float_format="%.5f", caption="asd", label="tab:delmh-orig-eca-single", escape=False)


def figures_and_tables_combined():
	pivot_I,pivot_S = [],[]
	pivot_plot_files = [pd.read_json('PivotBi/PivotBiCluster-result%d.json'%(x)) for x in range(4)]

	split_I,split_S = [],[]
	split_plot_files = [pd.read_json('SplitMerge/Split-Merge-result%d.json'%(x)) for x in range(4)]

	evolve_I,evolve_S = [],[]
	evolve_plot_files = [pd.read_csv('EvolveCluster/%d.csv'%(x)) for x in range(5)]
	id_fixer = [[(0,0),(1,1),(2,2)],
				[(0,5),(1,1),(2,2),(3,0),(4,4)],
				[(0,0),(1,1),(2,2),(3,3),(4,4)],
				[(0,0),(1,1),(2,2),(3,3),(4,4),(6,6)]]
	for i in range(4):
		a = evolve_plot_files[i+1].copy()
		evolve_plot_files[i]['cluster'] = evolve_plot_files[i]['cluster_ids']
		for j in range(len(id_fixer[i])):
			a.loc[a['cluster_ids'] == id_fixer[i][j][0], 'cluster_ids'] = id_fixer[i][j][1]
		a['cluster'] = a['cluster_ids']
		evolve_plot_files[i] = evolve_plot_files[i].append(a, ignore_index=True)

	del[evolve_plot_files[4]]
	plot_files = [evolve_plot_files,split_plot_files,pivot_plot_files]
	top_labels = ["D$_0$ - D$_1$","D$_1$ - D$_2$","D$_2$ - D$_3$","D$_3$ - D$_4$"]

	basefile = pd.read_csv('0.csv')
	for i in range(4):
		a = pd.read_csv('%d.csv'%(i+1))
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)

	columns=[str(x) for x in range(24)]

	for i in range(4):
		df = pivot_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		D = calculate_distances(df[columns])
		pivot_I.append(IC_av(pd.DataFrame(D), df["cluster"])[0])
		try:
			pivot_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			pivot_S.append(-11)
	data = [pivot_S, pivot_I]
	table_df = pd.DataFrame(data, index=["Silhouette", "IC-av"], columns=top_labels)
	table_df.to_latex(open("PivotBi_combined.tex", "w"), float_format="%.5f", caption="asd", label="tab:delmh-orig-pivotbi-combined", escape=False)
	for i in range(4):
		df = split_plot_files[i]
		df = df.merge(basefile, on='_id', how='inner')
		D = calculate_distances(df[columns])
		split_I.append(IC_av(pd.DataFrame(D), df["cluster"])[0])
		try:
			split_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			split_S.append(-11)
	data = [split_S, split_I]
	table_df = pd.DataFrame(data, index=["Silhouette", "IC-av"], columns=top_labels)
	table_df.to_latex(open("SplitMerge_combined.tex", "w"), float_format="%.5f", caption="asd", label="tab:delmh-orig-splitmerge-combined", escape=False)
	for i in range(4):
		df = evolve_plot_files[i]
		df['cluster'] = df['cluster_ids']
		#df = df.merge(basefile, on='_id', how='inner')
		D = calculate_distances(df[columns])
		evolve_I.append(IC_av(pd.DataFrame(D), df["cluster"])[0])
		try:
			evolve_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			evolve_S.append(-11)
	data = [evolve_S, evolve_I]
	table_df = pd.DataFrame(data, index=["Silhouette", "IC-av"], columns=top_labels)
	table_df.to_latex(open("EvolveCluster_combined.tex", "w"), float_format="%.5f", caption="asd", label="tab:delmh-orig-eca-combined", escape=False)

def figures_and_tables_combined_evolve_all_merged():
	evolve_I,evolve_S = [],[]
	evolve_plot_files = [pd.read_csv('EvolveCluster/%d.csv'%(x)) for x in range(5)]
	id_fixer = [[(0,0),(1,1),(2,2)],
				[(0,0),(1,1),(2,2),(3,0),(4,2)],
				[(0,0),(1,1),(2,2),(3,3),(4,4)],
				[(0,0),(1,1),(2,2),(3,3),(4,4),(6,0)]]
	for i in range(4):
		a = evolve_plot_files[i+1].copy()
		evolve_plot_files[i]['cluster'] = evolve_plot_files[i]['cluster_ids']
		for j in range(len(id_fixer[i])):
			a.loc[a['cluster_ids'] == id_fixer[i][j][0], 'cluster_ids'] = id_fixer[i][j][1]
		a['cluster'] = a['cluster_ids']
		evolve_plot_files[i] = evolve_plot_files[i].append(a, ignore_index=True)

	del[evolve_plot_files[4]]

	top_labels = ["D$_0$ - D$_1$","D$_1$ - D$_2$","D$_2$ - D$_3$","D$_3$ - D$_4$"]
	#plot_combined(plot_files, 'original_combined.pdf', top_labels)
	basefile = pd.read_csv('0.csv')
	for i in range(4):
		a = pd.read_csv('%d.csv'%(i+1))
		basefile = basefile.append(a, ignore_index=True)

	if(basefile['cluster'].min() != 0):
		basefile['cluster'] -= 1
	basefile.rename(columns={'cluster':'actual_cluster'}, inplace=True)
	columns=[str(x) for x in range(24)]
	for i in range(4):
		df = evolve_plot_files[i]
		#df = df.merge(basefile, on='_id', how='inner')
		D = calculate_distances(df[columns])
		evolve_I.append(IC_av(pd.DataFrame(D), df["cluster"])[0])
		try:
			evolve_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
		except ValueError:
			evolve_S.append(-11)
	data = [evolve_S, evolve_I]
	table_df = pd.DataFrame(data, index=["Silhouette", "IC-av"], columns=top_labels)
	table_df.to_latex(open("EvolveCluster_combined_all_merged.tex", "w"), float_format="%.5f", caption="asd", label="tab:delmh-orig-eca-combined", escape=False)




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

def IC_av(distance_matrix = None, labels = None):
    """Calculates Intra Cluster distance average of a clustering solution.

    Args:
        distance_matrix: 2-dimensional (n * n) matrix of Pandas Dataframe type containing distances between all nodes.
        labels: Clustering labels to assign each node.
    Returns:
        List containing total IC_av distance and each individual IC_av distance for each cluster.
    Raises:
        TypeError: if distance_matrix is not a Pandas.Dataframe.
        ValueError: if distance_matrix or labels is of None value.
    """
    if distance_matrix is None:
        raise ValueError("distance_matrix cannot be None value")
    if not isinstance(distance_matrix, pd.DataFrame):
        raise TypeError("distance_matrix must be DataFrame")
    if labels is None:
        raise ValueError("labels cannot be None value")


    # MST of the distance matrix
    mst = minimum_spanning_tree(distance_matrix.values)

    MED_matrix = pd.DataFrame(fast_IC_av(mst))
    
    IC_score = 0
    IC_scores = [0]
    clusters = list(sorted(set(labels))) # Get clusterlabels from labels

    MED_matrix.reset_index
    MED_matrix['clusters'] = labels


    # Traverse the mst for the largest edge and calculate the IC_av score
    for cluster in clusters:
        nodes = MED_matrix.loc[MED_matrix['clusters'] == cluster].index.tolist()
        total = 0
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                total += MED_matrix[i][j]
                #total += MED_matrix[i][j]**2

        total /= len(nodes)
        IC_score += total
        IC_scores.append(total)

    # Returns total IC_av score and individual IC_av scores for each cluster
    IC_scores[0] = IC_score
    return IC_scores

def d_med(tree, i, j):
    # traverse from i to j and return the max edge value
    return traverse(tree, i, i, j)

def traverse(tree, current, parent, target):
    if current == target:   # base case
        return tree[current][parent]

    # Get indexes that connect this node to the others except the parent
    children = [x for x in range(len(tree[current])) if tree[current][x] != 0 and x != parent]

    for child in children:
        dist = traverse(tree, child, current, target)
        if dist > 0:
            if dist > tree[current][parent]:
                return dist
            return tree[current][parent]

    return -1

def fast_IC_av(mst):
    """Returns the MED distance matrix for all nodes in the graph"""

    N = mst.shape[0]

    global graph
    graph = mst.toarray()
    
    # Preparation
    # for a non-directed graph, we need to symmetrize the distances
    # this is necessary since the minimum spanning tree only delivers edges in one direction
    for i in range(N):
        for j in range(i + 1, N):
            if graph[j, i] >= graph[i, j]:
                graph[i, j] = graph[j, i]
            else:
                graph[j, i] = graph[i, j]


    # 1) find all leaf nodes
    global leaf_list
    leaf_list = find_leaves(graph)
    # print("Leaves", leaf_list)

    # Set all distances with no direct connection to infinity
    graph[np.where(graph == 0)] = np.nan

    # Graph[i,i] should be zero
    graph.flat[::N + 1] = 0
    
    # 2) Store the results
    # MED_matrix saves the MED from node a to b 
    # MED is positive and symmetrical
    # Add takes the initial distances from graph
    global MED_matrix
    MED_matrix = copy.deepcopy(graph)
    

    # 3) Create a path dictionary to store started paths
    global path_list
    path_list = {}
    
    # Follow the paths while there are still leaf nodes
    while(leaf_list):
        first_node = leaf_list.pop()
        current_path = [first_node]

        neighbor_list = find_neighbors(graph, first_node, current_path)

        # since that is a leaf there should be only one neighbor
        assert len(neighbor_list) == 1

        # Calculate the distance to the next node
        dist = graph[first_node, neighbor_list[0]]

        follow_path(next_node=neighbor_list[0], current_path=current_path, dist=dist)

    return MED_matrix


def follow_path(next_node, current_path, dist):

    # Update the current path with the distance to the next node
    update_distances_node_centered(dist=dist, visited_list=current_path, shape=graph.shape[0], next_node=next_node)

    # Add the next element to the current path
    current_path.append(next_node)

    # Find path to the next elements
    neighbor_list = find_neighbors(graph, next_node, current_path)

    # Check whether an intersection or a leaf
    if not neighbor_list:
        # Remove the leaf from the leaf-list
        # Since it doesn't have to be a leaf it could be also stuck somewhere between known parts
        if next_node in leaf_list:
            leaf_list.remove(next_node)
        return

    elif len(neighbor_list) == 1:
        dist = graph[next_node, neighbor_list[0]]
        follow_path(next_node=neighbor_list[0], current_path=current_path, dist=dist)
    else:
        # Intersection case
        # Check if the current node is already in the dictionary
        if path_list.get(next_node):
            # Print already stored
            # Join lists and follow the path
            current_path = current_path + path_list.get(next_node)

            # Find the next way which is not in the joined list
            dest_node = [x for x in neighbor_list if x not in current_path]

            if not dest_node:
                # No more paths so the algorithm is finished
                print("No way to go. Empty join result")
                return
            elif len(dest_node) > 1:
                # Store the current path in the dictionary
                path_list.update({next_node:current_path})
                # Start again at the next leaf
                return

            else:
                # Follow the path with the now extended list
                # With the only left node to go
                dist = graph[next_node, dest_node[0]]
                follow_path(next_node=dest_node[0], current_path=current_path, dist=dist)

        else:
            # Store the current path in the dictionary
            path_list.update({next_node:current_path})
            # Start again at the next leaf
            return


def update_distances(dist, visited_list, shape):
    for node in range(shape):
        # Do not update distances in the already visited paths
        if node in visited_list:
            continue

        # Go through all the visited nodes and compare the distances, 
        # in case it is bigger perform an update
        for vis_node in visited_list:
            MED_matrix[vis_node, node] = np.nanmax([MED_matrix[vis_node, node], dist])
            MED_matrix[node, vis_node] = MED_matrix[vis_node, node]

def update_distances_node_centered(dist, visited_list, shape, next_node):
    # Going through the already visited nodes
    for vis_node in visited_list:
        # In case we already have a bigger distance to the new added node
        # we can skip this node since nothing will change
        # print(MED_matrix[vis_node, next_node], dist)
        if MED_matrix[vis_node, next_node] <= dist or np.isnan(MED_matrix[vis_node, next_node]):
            # Go through all nodes and update the distance
            for node in range(shape):
                # Do not update distances in the already visited paths
                if node in visited_list:
                    continue
                MED_matrix[vis_node, node] = np.nanmax([MED_matrix[vis_node, node], dist])
                MED_matrix[node, vis_node] = MED_matrix[vis_node, node]
        #else:
        #    print(next_node, visited_list, dist, vis_node)
        #    print(MED_matrix)

        # Go through all the visited nodes and compare the distances, 
        # in case it is bigger perform an update


def find_neighbors(graph, node, current_path):
    neighbor_list = []
    for ii in range(graph.shape[0]):
        if not np.isnan(graph[node, ii]) and ii not in current_path:
            neighbor_list.append(ii)

    return neighbor_list
    

def find_leaves(graph):
    """Returns a list of all leaves"""
    
    leaf_list = []

    for row in range(graph.shape[0]):
        # print(np.count_nonzero(graph[row]))
        # one non-zero element means that there is only one connection to another node
        # making it to a leaf
        if np.count_nonzero(graph[row]) == 1:
            leaf_list.append(row)

        # Iris data set 54 leaves, so more than one third
    return leaf_list

def main():
	figures_and_tables_single()
	figures_and_tables_combined()
	figures_and_tables_combined_evolve_all_merged()

if __name__ == '__main__':
	main()
