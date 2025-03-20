
import pandas
from matplotlib import pyplot as plt
import numpy as np

from sklearn import metrics

# each row is an image vector length 28^2; can visualize by reshaping if needed.
df = pandas.read_csv('data/train.csv') # Half the MNIST data.

fig,ax = plt.subplots(1,3, figsize=(9,3), sharex=True, sharey=True)

ax[0].imshow( np.reshape(df.iloc[500,1:].values, (28,28))  )
ax[1].imshow( np.reshape(df.iloc[3,1:].values, (28,28))  )
ax[2].imshow( np.reshape(df.iloc[4,1:].values, (28,28))  )

fig.show()

# pull out the data for only 4s and 9s; see if optimal transport can 
# associate images.

#
# pretend you don't know what's a 4 and what's a 9. building a graph 
# connecting the images to go from vertex i to vertex j working off 
# of the matrix T. (??????????)

X = df.iloc[:100, 1:] # ignore column 0; take first 100 data points

D = metrics.pairwise_distances(X)
D[D> 3000] = np.inf
