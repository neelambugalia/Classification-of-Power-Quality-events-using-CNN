
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
import numpy as np
import math

x1=np.linspace(0,0.12,768)
noise = np.random.normal(0, 1,768)
t=(2/3)*(np.pi)
w1=2*np.pi*50
Fm=20  ## flicker magnitude
ff=10 ## flicker frequency
wf=2*np.pi*ff
Tm=1.4*np.sqrt(2)*230 ## transient magnitude
t1=0 ## transient starting time
e1=0.00009  ## will use this to vary transient starting time
wosc=2*np.pi*400  ## angular oscillator frequncy of transient
ts=0.005  ## transient settling time
N=1
Z=[0]*N
#ts =settling time
p=math.log(0.0001)/(-ts)

for i in range(N):
	wf=2*np.pi*(ff+e1*i)
	x2=np.linspace(0,t1+i*e1,int(6400*(t1+i*e1)))
	x3=np.linspace(t1+i*e1,ts+t1+i*e1,int(6400*ts))
	x4=np.linspace(ts+t1+i*e1,0.12,int(6400*(0.12-t1-i*e1-ts)))
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
	
	Y=[]
	Y.append(X1)
	Y.append(X2)
	Y.append(X3)


	X= Y
	rp = RecurrencePlot(threshold='point', percentage=20)
	X_rp = rp.fit_transform(X)
	fig,ax = plt.subplots(1)
	fig=plt.figure(figsize=(6, 6))
# Show the results for the first time series
	plt.figure(figsize=(6, 6))
	plt.imshow(X_rp[0], cmap='binary', origin='lower')
	title='FLICKERS AND TRANSIENTS'
	plt.title(title, fontsize=18)
	#plt.show()
	save_name =  'RCP/FT.png'
	plt.savefig(save_name)
	plt.close(fig)


