import numpy as np
import matplotlib.pyplot as plt

import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt

N= 1

prefix = "try/out1_"
save_pre = 'scalograms/out1/'

for i in range(N):
	file_name = prefix + str(i) + '.wav'


	st = obspy.read(file_name)
	tr = st[0]
	npts = tr.stats.npts
	dt = tr.stats.delta
	t = np.linspace(0, dt * npts, npts)
	f_min = 30
	f_max = 75
	w0=10

	scalogram = cwt(tr.data, dt, w0, f_min, f_max)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	x, y = np.meshgrid(t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
	
	ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
	ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
	ax.set_ylabel("Frequency [Hz]")
	ax.set_yscale('log')
	ax.set_ylim(f_min, f_max)

	save_name = save_pre + str(i) + '.png'
	# plt.savefig(save_name)
	# plt.close(fig)
	plt.show()