import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
#from pyts.datasets import load_gunpoint
import pandas as pd

for j in range(3):
    filename="out" +str(j)+"_.csv"
    df = pd.read_csv(filename)
    N=15
    save_pre = 'trial/GAFD/' +str(j) +'/'
    for i in range(N):
        col_name = "col_" +str(i)
        X = df[col_name].values
        X = [X]


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
        titles = ['Gramian Angular difference Field']
        for image, title, ax in zip(images, titles, grid):
            im = ax.imshow(image, cmap='rainbow', origin='lower')
            ax.set_title(title, fontdict={'fontsize': 16})
        #ax.cax.colorbar(im)
        ax.cax.toggle_label(False)

        fig=plt.suptitle('', y=0.92, fontsize=20)

        save_name = save_pre +str(j)+'_'+str(i) + '.png'
        plt.savefig(save_name)
        #plt.close(fig)
        
#plt.show()