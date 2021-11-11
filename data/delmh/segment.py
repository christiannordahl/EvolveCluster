import pandas as pd

files = ["1222.csv","1200.csv"]
segments = [0, 0.4, 0.55, 0.70, 0.85, 1]

for file in files:
	df = pd.read_csv('initial_tests/profilers/%s'%(file), index_col=0, parse_dates=True)
	for i in range(len(segments)-1):
		a = df.iloc[int(segments[i]*len(df)):int(segments[i+1]*len(df)),:]
		a.to_csv('initial_tests/profilers/segmented/%s_%d.csv'%(file.split('.')[0], i))
