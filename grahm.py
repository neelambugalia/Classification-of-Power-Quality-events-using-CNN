import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
#from pyts.datasets import load_gunpoint
import pandas as pd
import numpy as np 

for j in range(3):
    filename="out" +str(j)+".csv"
    df = pd.read_csv(filename)
    N=1016
    
    for i in range(N):
        col_name = "col_" +str(i)
        Y = df[col_name].values
        X1 = Y[2:768]
        X2=Y[770:1536]
        X3=Y[1538:2304]
        X=[]
        X.append(X1)
        X.append(X2)
        X.append(X3)
        X = np.array(X, dtype = object)
        # print(X.dtype, X.shape)
        # X.reshape(1, -1)
        # print(X)

    # Transform the time series into Gramian Angular Fields
        #gasf = GramianAngularField(image_size=24, method='summation')
        #X_gasf = gasf.fit_transform(X)
        gadf = GramianAngularField(image_size=24, method='difference')
        X_gadf = gadf.fit_transform(X)

    # Show the images for the first time series
        fig = plt.figure(figsize=(6, 6))
        grid = ImageGrid(fig,111,
                 nrows_ncols=(1, 1),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.3,
                 )
        images = [X_gadf[0]]
        titles = []
        for image, title, ax in zip(images, titles, grid):
            im = ax.imshow(image, cmap='rainbow', origin='lower')
            ax.set_title(title, fontdict={'fontsize': 16})
        #ax.cax.colorbar(im)
        ax.cax.toggle_label(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])


        fig=plt.suptitle('', y=0.92, fontsize=20)
        if(j==1):
            save_pre = 'GADF/normal case/'
            save_name = save_pre+'img'+str(i) + '.png'
        if(j==0):
            save_pre = 'GADF/abnormal frequency/'
            save_name = save_pre+'img'+str(i+N) + '.png'
        if(j==2):
            save_pre = 'GADF/abnormal frequency/'
            save_name = save_pre+'img'+str(i+2*N) + '.png'


        plt.savefig(save_name)
        #plt.close(fig)

        
#plt.show()