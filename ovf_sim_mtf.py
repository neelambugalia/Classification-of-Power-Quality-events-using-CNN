import matplotlib.pyplot as plt
import numpy as np
from pyts.image import MarkovTransitionField
import pandas as pd

x1=np.linspace(0,0.12,768)
noise = np.random.normal(0, 1,768)
t=(2/3)*(np.pi)
e1=0.0004
e2=0.095
w1=2*np.pi*50.3

e3=0.0009
e4=0.0015

N=1148
X=[0]*N
a=[]


for i in range(164):
	X1= np.sqrt(2)*(250+i*e2)*np.sin(i*e1+w1*x1)
	X2= np.sqrt(2)*(230-e3*i)*np.sin(i*e1+w1*x1+t)
	X3= np.sqrt(2)*(230-e3*i)*np.sin(i*e1+w1*x1-t)
	X12=X1+noise
	X23=X2+noise
	X31=X3+noise
	Y=[]
	Y.append(X12)
	Y.append(X23)
	Y.append(X31)
	X[i]= Y

for i in range(164,328,1):
	X1=np.sqrt(2)*(230-e3*i)*np.sin((i-144)*e1+w1*x1)
	X2=np.sqrt(2)*(250+i*e2)*np.sin((i-144)*e1+w1*x1+t)
	X3=np.sqrt(2)*(230-e3*i)*np.sin((i-144)*e1+w1*x1-t)
	X12=X1+noise
	X23=X2+noise
	X31=X3+noise
	Y=[]
	Y.append(X12)
	Y.append(X23)
	Y.append(X31)
	X[i]= Y

for i in range(328,492,1):
	X1= np.sqrt(2)*(230-e3*i)*np.sin((i-288)*e1+w1*x1)
	X2= np.sqrt(2)*(230-e3*i)*np.sin((i-288)*e1+w1*x1+t)
	X3= np.sqrt(2)*(250+i*e2)*np.sin((i-288)*e1+w1*x1-t)
	X12=X1+noise
	X23=X2+noise
	X31=X3+noise
	Y=[]
	Y.append(X12)
	Y.append(X23)
	Y.append(X31)
	X[i]= Y

for i in range(492,656,1):
	X1=np.sqrt(2)*(250+i*e2)*np.sin((i-432)*e1+w1*x1)
	X2=np.sqrt(2)*(250+i*e2)*np.sin((i-432)*e1+w1*x1+t)
	X3=np.sqrt(2)*(230-e4*i)*np.sin((i-432)*e1+w1*x1-t)
	X12=X1+noise
	X23=X2+noise
	X31=X3+noise
	Y=[]
	Y.append(X12)
	Y.append(X23)
	Y.append(X31)
	X[i]= Y
	
for i in range(656,820,1):
	X1= np.sqrt(2)*(250+i*e2)*np.sin((i-576)*e1+w1*x1)
	X2= np.sqrt(2)*(230-e4*i)*np.sin((i-576)*e1+w1*x1+t)
	X3= np.sqrt(2)*(250+i*e2)*np.sin((i-576)*e1+w1*x1-t)
	X12=X1+noise
	X23=X2+noise
	X31=X3+noise
	Y=[]
	Y.append(X12)
	Y.append(X23)
	Y.append(X31)
	X[i]= Y

for i in range(820,984,1):
	X1= np.sqrt(2)*(230-e4*i)*np.sin((i-720)*e1+w1*x1)
	X2= np.sqrt(2)*(250+i*e2)*np.sin((i-720)*e1+w1*x1+t)
	X3= np.sqrt(2)*(250+i*e2)*np.sin((i-720)*e1+w1*x1-t)
	X12=X1+noise
	X23=X2+noise
	X31=X3+noise
	Y=[]
	Y.append(X12)
	Y.append(X23)
	Y.append(X31)
	X[i]= Y

for i in range(984,1148,1):
	X1= np.sqrt(2)*(250+i*e2)*np.sin((i-864)*e1+w1*x1)
	X2= np.sqrt(2)*(250+i*e2)*np.sin((i-864)*e1+w1*x1+t)
	X3= np.sqrt(2)*(250+i*e2)*np.sin((i-864)*e1+w1*x1-t)
	X12=X1+noise
	X23=X2+noise
	X31=X3+noise
	Y=[]
	Y.append(X12)
	Y.append(X23)
	Y.append(X31)
	X[i]= Y


for i in range(	N):
		# MTF transformation
	mtf = MarkovTransitionField(image_size=24)
	X_mtf = mtf.fit_transform(X[i])

		# Show the image for the first time series
	fig,ax = plt.subplots(1)
	fig=plt.figure(figsize=(6, 6))
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.gca().axes.get_xaxis().set_visible(False)
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	
	plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
	
	#plt.title('', fontsize=18)
	#plt.colorbar(fraction=0.0457, pad=0.04)

	save_name =  'OVF/'+'ovf_'+str(i) + '.png'
	plt.savefig(save_name)
	plt.close(fig)
	#plt.show()
