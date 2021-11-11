import pandas as pd
import json

houses = ['1200','1222']
best_solutions = 	[[6,4,3,3,3],
					 [3,3,4,3,2]]
_id = 0
i=0
for house in houses:
	j=0
	files = ['%s_%d.csv'%(house, x) for x in range(5)]
	info_path = 'initial_clusters/'
	data_path = 'orig/'
	session = {}
	for file in files:
		info = pd.read_csv(info_path+file, index_col=0)
		data = pd.read_csv(data_path+file, index_col=0, parse_dates=True)

		clusters = json.loads(info.loc[best_solutions[i][j]]['sil_clusters'])
		centroids = sorted(list(set(clusters)))
		cluster_ids = [None for x in range(len(clusters))]
		for k in range(len(centroids)):
			for l in range(len(clusters)):
				if clusters[l] == centroids[k]:
					cluster_ids[l] = k

		ids = []
		attributes = []
		for k in range(len(data)):
			ids.append(_id)
			_id+=1
			attributes.append(data.iloc[k,:].to_list())

		new_data = pd.DataFrame()
		new_data['attributes'] = attributes
		new_data['cluster'] = cluster_ids
		new_data['_id'] = ids

		data['cluster'] = cluster_ids
		data['_id'] = ids
		data.to_csv(file)

		C = {}
		for k in range(len(centroids)):
			C[str(k)] = attributes[k]

		if j < len(files)-1:
			new_data.to_json(file.split('_')[0]+'/firstSet%d.json'%(j), orient='records')
			session['firstSet%d'%(j)] = C
		if j > 0:
			new_data.to_json(file.split('_')[0]+'/secondSet%d.json'%(j-1), orient='records')
			session['secondSet%d'%(j-1)] = C
		j+=1

	session = pd.DataFrame.from_dict(session)
	session.to_json(file.split('_')[0]+'/session_terms.json')
	i += 1
