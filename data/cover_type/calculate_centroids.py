import numpy as np
import pandas as pd
import json
from sklearn.metrics import pairwise_distances

centroids = [
	[0.7398903174,-0.0506073969,-0.0843244657,0.2803762515,0.081890998,0.4892927425,-0.0981105269,-0.0328796657,0.0945401496,-0.2378490399,0.28850927,0.0787680904,-0.1076096347,-0.3211603342],
	[0.0060832036,-0.0373268065,-0.202171002,-0.0378943722,-0.1337830244,0.1912789421,0.0880684882,0.1393381432,0.0370364165,0.5169032092,0.464019848,-0.0729721711,-0.2863848584,-0.3187764269],
	[-1.5440905279,0.3093470263,1.011205655,-0.1391179746,0.4880663397,-1.1715800504,-0.6099499397,-0.4113982748,0.0774899758,-1.3687915813,-1.8273335195,-0.1004022133,0.8204028711,1.7393026962],
	[-2.1729927413,-0.0116268048,0.7083770323,-0.6735601733,-0.0087775428,-1.1932001376,0.4671784546,-0.3441013463,-0.7373623087,-1.3863465284,-1.8273335195,-0.1004022133,-0.3813779367,3.1137095506],
	[-0.2466309784,-0.0782747471,0.3861350858,-0.1882944978,0.1523626643,-1.0019528981,0.2954136237,-0.2439250662,-0.45517844,-0.8368298823,-0.7395772735,-0.1004022133,1.2462395612,-0.3211603342],
	[-1.4418532863,0.3947318067,0.7929025333,-0.3930206514,0.1260563155,-1.1105153268,-0.9759832428,-0.7420255591,0.2895992994,-1.2873622691,-1.8273335195,-0.1004022133,0.980582043,1.556115088],
	[1.8648455365,0.1344677932,0.1487126724,0.4708363429,0.534186793,-0.2904170386,0.0094845174,-0.0640790709,-0.0663175972,-0.7886679144,-1.2162286312,0.9957446771,1.5218770918,-0.3211603342]
	]

df = pd.read_csv('0.csv', index_col=0)
nearest_data_points = []
for i in range(len(centroids)):
	ids = df[df['cluster'] == i+1]['_id']
	ids = ids.reset_index(drop=True)

	dists = pairwise_distances(np.array(centroids[i]).reshape(1,-1), df[df['cluster'] == i+1].iloc[:,:14])
	index = np.argmin(dists)

	nearest_data_points.append(ids[index])

C = {}
for i in range(1,8):
	C[i-1] = df[df['cluster']==i]['_id'].index.to_list()

medoids = df[df['_id'].isin(nearest_data_points)].index.to_list()

with open('initial_clusters.json','w') as f:
	f.write(json.dumps(C))
df = pd.DataFrame()
df['medoids'] = medoids
df.to_csv('medoids.csv',index=False)

with open('initial_clusters.json','r') as f:
	test = json.load(f)
df = pd.read_csv('medoids.csv',index_col=None)
