import numpy as np
import pandas as pd
import copy

from scipy.spatial.distance import euclidean
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.metrics import pairwise_distances

def kmedoids(D, k, medoids=None, tmax=100):
    '''
        D is the dissimilarity matrix of the data to be clustered
        in a n x n format. 
        
        k is the amount of clusters.
        
        medoids is the initial medoids chosen for the algorithm,
        if none is passed we initialize with random ones below
        
        tmax is the maximum amount of iterations for the alg
    '''
    # Dimensions of dissimilarity matrix
    m,n = D.shape
    
    # Random initation of medoid indices if none is passed
    if medoids is None:
        medoids = np.sort(np.random.choice(n, k))
    new_medoids = medoids.copy()
    medoids_copy = medoids.copy()
    np.sort(medoids_copy)
    
    # Initial dict for our clusters
    C = {}
    
    for t in range(tmax):
        # Determine the clusters
        J = np.argmin(D[:,medoids], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        
        # Update the cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])], axis=1)
            j = np.argmin(J)
            new_medoids[kappa] = C[kappa][j]
        new_medoids_copy = new_medoids.copy()
        np.sort(new_medoids_copy)
        
        # check for convergence, i.e. are the matrices the same?
        # if so, then we are finished
        if np.array_equal(medoids_copy, new_medoids_copy):
            break
            
        medoids = new_medoids.copy()
    
    # If we didn't break the loop above, then we have to update the
    # cluster memberships again
    else:
        J = np.argmin(D[:,medoids], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    
    return C, medoids

def calculate_distances(data):
    #dists = [[euclidean(data[x], data[y]) for y in range(len(data))] for x in range(len(data))]
    #return np.array(dists)
    #z = np.array([[complex(c[0], c[1]) for c in data]]) # notice the [[ ... ]]
    #return abs(z.T-z)
    return pairwise_distances(data,data)


# IC_av
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

# Connectivity
def find_nearest_neighbors(sample, neighbors, n_neighbors):
    """Finds the nearest neighbors to the given sample"""
    neighbors = np.delete(neighbors, sample)
    nearest_neibour_indexes = np.argsort(neighbors)[:n_neighbors]
    return nearest_neibour_indexes

def get_connectivity(dataframe, i, j, j_index):
    """Gets the connectivity value from a given dataframe"""
    i_class = dataframe.iloc[i]["class"]
    j_class = dataframe.iloc[j]["class"]
    state = (0 if i_class == j_class else float(1)/(j_index + 1))
    return state

def connectivity_samples(X, y, n_neighbors, metric=None, distance_matrix=None, **kwds):
    """Calculates the connectivity for each sample in the dataset."""
    dataframe = pd.DataFrame(data=X, index=range(len(X)))
    dataframe["class"] = y
    N = len(X)
    
    if distance_matrix is None:
        distance_matrix = pairwise_distances(X, metric=metric, **kwds)
    
    connectivity_list = []
    for i in range(N):
        connectivity_sum = 0
        nearest_neighbors = find_nearest_neighbors(i, distance_matrix[i], n_neighbors)
        for j in range(n_neighbors):
            connectivity_sum += get_connectivity(dataframe, i, nearest_neighbors[j], j)
        connectivity_list.append(connectivity_sum)
    return connectivity_list

def calculate_connectivity(X_train, y_train, columns, n_neighbors, metric=None, distance_matrix=None):
    si_samples = connectivity_samples(X_train, y_train, n_neighbors, metric, distance_matrix)
    dataframe = pd.DataFrame(data=X_train, columns=columns, index=range(0, len(y_train)))
    dataframe["CONN"] = si_samples
    dataframe["class"] = y_train
    #sorted_dataframe = dataframe.sort_values(["CONN"], ascending=[True])
    return dataframe
