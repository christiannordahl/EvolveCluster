import pandas as pd
import numpy as np 

files = ["1200.csv", "1222.csv"]
data = [pd.read_csv('split/profilers/%s'%(file)) for file in files]
cols = ['RecorderID','ProfileID','Valid']

for d,f in zip(data, files):
	d['Datefield'] = pd.to_datetime(d['Datefield'])
	d.drop(columns=cols, inplace=True)
	start = d.iloc[0, 0]
	start = start.replace(hour=0, minute=0)
	end = d.iloc[-1, 0]
	end = end.replace(hour=0, minute=0)
	
	# Remove leading and trailing data, i.e. remove partial days
	d.drop(d[(d['Datefield'].dt.day == start.day) & (d['Datefield'].dt.month == start.month) & (d['Datefield'].dt.year == start.year)].index, inplace=True)
	d.drop(d[(d['Datefield'].dt.day == end.day) & (d['Datefield'].dt.month == end.month) & (d['Datefield'].dt.year == end.year)].index, inplace=True)
	d.reset_index(drop=True, inplace=True)
	
	# Fix the index and the missing values, i.e. reindex by dates and fill non-existing
	# timestamps with np.nan
	start += pd.DateOffset(days=1)
	end -= pd.DateOffset(minutes=1)
	idx = pd.date_range(start, end, freq='H')
	d.index = d['Datefield']
	del d['Datefield']
	d = d.reindex(idx, fill_value=np.nan)

	idx = pd.date_range(start, end, freq='D')
	new_df = pd.DataFrame(d.values.reshape(-1, 24), index=idx)
	removed_rows = new_df[new_df.isnull().any(axis=1)]
	new_df.dropna(inplace=True)

	new_df.to_csv("initial_tests/profilers/%s"%(f))
	removed_rows.to_csv("initial_tests/profilers/removed_%s"%(f))
