import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax = plt.axes()
df = pd.read_csv('out0.csv')
col_name = "col_952"
X=np.linspace(0,0.36,2304)
fig=plt.plot(X,df)
#plt.show()
save_name = 'phaseplot/' + 'plt'+'.png'
plt.savefig(save_name)