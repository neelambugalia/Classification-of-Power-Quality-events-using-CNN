import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1,figsize=(12,12))
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection="3d")
df = pd.read_csv('29.csv')


Y1 = df['phase1'].values
Y2 = df['phase2'].values
Y3 = df['phase3'].values
X1 = Y1[258:1030]
X2 = Y2[258:1030]
X3 = Y3[258:1030]

fig=plt.plot(X1,X2,X3)
plt.show()
save_name = 'phaseplot3d'+'.png'
plt.savefig(save_name)