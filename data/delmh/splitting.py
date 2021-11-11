import pandas as pd

files = [	"1994_A.csv","1995_A.csv",
			"1996_A.csv","1997_A.csv",
			"1998_A.csv","1999_A.csv",
			"2000_A.csv","2001_A.csv",
			"2002_A.csv","2003_A.csv",
			"2004_A.csv","2005_A.csv",
			"2006_A.csv","2007_A.csv",
			"2008_A.csv","2009_A.csv",
			"2010_A.csv","2011_A.csv",
			"2012_A.csv","2013_A.csv",
			"2014_A.csv"]

dfs = {}
recorder_dfs = {}
for file in files:
	print("Reading file:", file)
	df = pd.read_csv("original/"+file)
	df['Datefield'] = pd.to_datetime(df['Datefield'])

	profilers = df['ProfileID'].unique()
	num_profilers = len(profilers)
	for profiler in profilers:
		if profiler not in dfs:
			dfs[profiler] = pd.DataFrame(columns=df.columns.tolist())

		dfs[profiler] = dfs[profiler].append(df[df['ProfileID'] == profiler], ignore_index=True)


for key in dfs:
	dfs[key].sort_values(by='Datefield').to_csv("split/profilers/%d.csv"%(key),index=False)
