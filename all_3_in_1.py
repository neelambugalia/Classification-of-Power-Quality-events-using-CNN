import matplotlib.pyplot as plt
import numpy as np
from pyts.image import MarkovTransitionField
import pandas as pd

N=1
data=[0]*N
for k in range(N):
	file_name= str(k)+'.csv'
	f =open(file_name,'r')
	data[k] = f.read().split('\n')[277:-1]
	new_data1=[]
	new_data2=[]
	new_data3=[]
	m = len(data[k])
	for j in range(m):
		new_data1 += [float(data[k][j].split(',')[2][1:-1])]
		new_data2 += [float(data[k][j].split(',')[3][1:-1])]
		new_data3 += [float(data[k][j].split(',')[4][1:-1])]
	X=[]
	X.append(new_data1)
	X.append(new_data2)
	X.append(new_data3)
	mtf = MarkovTransitionField(image_size=24)
	X_mtf = mtf.fit_transform(X)
	fig,ax = plt.subplots(1)
	fig=plt.figure(figsize=(6, 6))
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.gca().axes.get_xaxis().set_visible(False)
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	
	plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
	
	#plt.title('', fontsize=18)
	#plt.colorbar(fraction=0.0457, pad=0.04)

	save_name =  'High Impulse/'+"High_impilse_"+str(k) + '.png'
	plt.savefig(save_name)
	plt.close(fig)

