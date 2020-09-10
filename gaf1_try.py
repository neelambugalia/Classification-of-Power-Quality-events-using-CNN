import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
import numpy as np
import pandas as pd
import math
from pyts.image import MarkovTransitionField

N=5
Z=[0]*N
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
	Z[k]=X

for i in range(	N):

		# MTF transformation
	mtf = MarkovTransitionField(image_size=24)
	X_mtf = mtf.fit_transform(Z[i])

	if(i==0):
		title='NORMAL CASE'
	if(i==1):
		title ='ABNORMAL FREQUENCY'
	if(i==2):
		title ='VOLTAGE SAG'
	if(i==3):
		title ='VOLTAGE SWELL'
	if(i==4):
		title='WAVEFORM DISTORTION'	
	fig,ax = plt.subplots(1)
	fig=plt.figure(figsize=(6, 6))
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.gca().axes.get_xaxis().set_visible(False)
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	plt.title(title)
	plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
	
	#plt.title('', fontsize=18)
	#plt.colorbar(fraction=0.0457, pad=0.04)

	save_name =  'MTF/'+title + '.png'
	plt.savefig(save_name)
	plt.close(fig)
	
	

        
#plt.show()