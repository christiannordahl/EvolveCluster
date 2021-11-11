import pandas as pd
import random

with open('s1.txt', 'r') as f, open('labels.txt', 'r') as l:	
	data = [list(map(int, line.strip().split())) for line in f.readlines()]
	labels = [int(line.strip()) for line in l.readlines()]
	data = pd.DataFrame(data, columns=['x','y'])
	# Normalizing data next row
	data = (data-data.min())/(data.max()-data.min())
	data['cluster'] = labels
	data['_id'] = data.index
	dat = [pd.DataFrame(columns=['x','y','cluster','_id']) for x in range(6)]

	# Continuous
	for i in range(1, 16, 1):
		tmp = data[data['cluster'] == i]

		dat[0] = dat[0].append(tmp[:int(len(tmp)*0.2)], ignore_index=True)
		dat[1] = dat[1].append(tmp[int(len(tmp)*0.2):int(len(tmp)*0.4)], ignore_index=True)
		dat[2] = dat[2].append(tmp[int(len(tmp)*0.4):int(len(tmp)*0.6)], ignore_index=True)
		dat[3] = dat[3].append(tmp[int(len(tmp)*0.6):int(len(tmp)*0.8)], ignore_index=True)
		dat[4] = dat[4].append(tmp[int(len(tmp)*0.8):int(len(tmp))], ignore_index=True)

	ranges = [0,0.2,0.4,0.6,0.8,1]
	#ranges = [0,0.5,0.6,0.7,0.8,0.9,1]

	for i in range(5):
		dat[i].to_csv('continuous/%d.csv'%(i), index=False)
		data[int(len(data)*ranges[i]):int(len(data)*ranges[i+1])].to_csv('original/%d.csv'%(i), index=False)

	attributes = []
	for i in range(len(data)):
		attributes.append([data.iloc[i,:2][0],data.iloc[i,:2][1]])
	
	data['attributes'] = attributes
	data = data[['attributes','cluster','_id']]

	for i in range(5):
		attributes = []
		for j in range(len(dat[i])):
			attributes.append([dat[i].iloc[j,:2][0],dat[i].iloc[j,:2][1]])
		dat[i]['attributes'] = attributes
		dat[i] = dat[i][['attributes','cluster','_id']]

	for i in range(5):
		dat[i].to_json('continuous/%d.json'%(i), orient='records')
		data[int(len(data)*ranges[i]):int(len(data)*ranges[i+1])].to_json('original/%d.json'%(i), orient='records')

	for i in range(1,5):
		dat[0] = dat[0].append(dat[i], ignore_index=True)

	dat[0].to_json('continuous/scaled_benchmark.json')
	data.to_json('original/scaled_benchmark.json')



