import math
import matplotlib.pyplot as plt
import numpy as np
from pyts.image import MarkovTransitionField
import pandas as pd

x1=np.linspace(0,0.12,768)
noise = np.random.normal(0, 1,768)
t=(2/3)*(np.pi)
w1=2*np.pi*50
Fm=20  ## flicker magnitude
ff=10 ## flicker frequency
wf=2*np.pi*ff
Tm=1.4*np.sqrt(2)*230 ## transient magnitude
t1=0 ## transient starting time
e1=0.0001  ## will use this to vary transient starting time
wosc=2*np.pi*400  ## angular oscillator frequncy of transient
p=0.015  ## transient settling time
N=1001
Z=[0]*N

for i in range(N):
	wf=2*np.pi*(ff+e1*i)
	x2=np.linspace(0,t1+i*e1,int(6400*(t1+i*e1)))
	x3=np.linspace(t1+i*e1,p+t1+i*e1,int(6400*p))
	x4=np.linspace(p+t1+i*e1,0.12,int(6400*(0.12-t1-i*e1-p))+1)
	X11= (np.sqrt(2)*230+Fm*np.sin(wf*x2))*np.sin(w1*x2)
	X12=(np.sqrt(2)*230+Fm*np.sin(wf*x3))*np.sin(w1*x3)+Tm*np.sin(wosc*x3)*math.exp(-p*(t1+i*e1))
	X13=(np.sqrt(2)*230+Fm*np.sin(wf*x4))*np.sin(w1*x4)
	
	X21= (np.sqrt(2)*230+Fm*np.sin(wf*x2+t))*np.sin(w1*x2+t)
	X22=(np.sqrt(2)*230+Fm*np.sin(wf*x3+t))*np.sin(w1*x3+t)+Tm*np.sin(wosc*x3)*math.exp(-p*(t1+i*e1))
	X23=(np.sqrt(2)*230+Fm*np.sin(wf*x4+t))*np.sin(w1*x4+t)
	
	X31= (np.sqrt(2)*230+Fm*np.sin(wf*x2-t))*np.sin(w1*x2-t)
	X32=(np.sqrt(2)*230+Fm*np.sin(wf*x3-t))*np.sin(w1*x3-t)+Tm*np.sin(wosc*x3)*math.exp(-p*(t1+i*e1))
	X33=(np.sqrt(2)*230+Fm*np.sin(wf*x4-t))*np.sin(w1*x4-t)
	
	X1=[]
	for j in X11:
		X1.append(j)
	for j in X12:
		X1.append(j)
	for j in X13:
		X1.append(j)
	X2=[]
	for j in X21:
		X2.append(j)
	for j in X22:
		X2.append(j)
	for j in X23:
		X2.append(j)
	X3=[]
	for j in X31:
		X3.append(j)
	for j in X32:
		X3.append(j)
	for j in X33:
		X3.append(j)
	n1=len(X1)
	n2=len(X2)
	n3=len(X3)
	n4=len(noise)
	for j in range (768):
		X1[j]=X1[j]+noise[j]
		X2[j]=X2[j]+noise[j]
		X3[j]=X3[j]+noise[j]

	
	Y=[]
	Y.append(X1)
	Y.append(X2)
	Y.append(X3)
	print(Y)
	Z[i]= Y
for i in range(	N):
	gadf = GramianAngularField(image_size=24, method='difference')
	X_gadf = gadf.fit_transform(Z[i])

	fig = plt.figure(figsize=(6, 6))
	grid = ImageGrid(fig,111,
            nrows_ncols=(1, 1),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="7%",
            cbar_pad=0.3,)
	images = [X_gadf[0]]
	titles = ['']
	for image, title, ax in zip(images, titles, grid):
		im = ax.imshow(image, cmap='rainbow', origin='lower')
		ax.set_title(title, fontdict={'fontsize': 16})
	ax.cax.toggle_label(True)
	fig=plt.suptitle('', y=0.92, fontsize=20)
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.gca().axes.get_xaxis().set_visible(False)
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	save_name = 'GADF/'+'ufv'+str(i) + '.png'
	plt.savefig(save_name)
