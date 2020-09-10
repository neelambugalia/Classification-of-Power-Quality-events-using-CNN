import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
import pandas as pd


for j in range(3):
	filename="out" +str(j)+"_.csv"
	df = pd.read_csv(filename)
	N=2
	for i in range(N):
		col_name = "col_" +str(i)
		X = df[col_name].values
		X = [X]
		# MTF transformation
		mtf = MarkovTransitionField(image_size=24)
		X_mtf = mtf.fit_transform(X)

		# Show the image for the first time series
		fig=plt.figure(figsize=(6, 6))

		plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        
        #fig=plt.suptitle('', y=0.92, fontsize=20)
        #plt.close(fig)

		plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
		plt.title('Markov Transition Field', fontsize=18)

		#plt.colorbar(fraction=0.0457, pad=0.04)
		if(j==1):
            save_pre = 'MTF/normal case/'
            save_name = save_pre+"mtf_snap"+str(i) + '.png'
        if(j==0):
            save_pre = 'MTF/abnormal frequency/'
            save_name = save_pre+'mtf_uf'+str(i) + '.png'
        if(j==2):
            save_pre = 'MTF/abnormal frequency/'
            save_name = save_pre+"mtf_of"+'img'+str(i) + '.png'
		plt.savefig(save_name)
		plt.close(fig)
	#plt.show()
