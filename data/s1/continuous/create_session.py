import pandas as pd

files = [x for x in range(5)]
final_b = {}
for i in range(5):
	f = open('%d.txt'%(i), 'r')
	df = pd.read_csv('%d.csv'%(i))


	a = f.readlines()[1].rstrip()
	a = [int(x) for x in a.split(",")]

	b = {}
	for j in range(len(a)):
		b[j] = [df.loc[a[j]]['x'], df.loc[a[j]]['y']]

	final_b[i] = b

b = pd.DataFrame.from_dict(final_b)
b.to_json('session_terms.json')

