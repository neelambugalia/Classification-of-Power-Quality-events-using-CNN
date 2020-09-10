import pandas as pd
import soundfile as sf

# assume we have columns 'time' and 'value'
df = pd.read_csv('0.csv')

n_entries = df.size
n_rows = len(df)

N = n_entries // n_rows

# print(N)
prefix = "WAV/out1/out1_"
for i in range(N):
	file_name = prefix + str(i) + '.wav'
	col_name = 'col_0' 
	data = df[col_name].values
	n = len(data)

	timespan = 0.12
	sample_rate_hz = int(n / timespan)

	sf.write(file_name, data, sample_rate_hz)

