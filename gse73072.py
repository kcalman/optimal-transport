import pandas
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition

# file download:
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73072/matrix/GSE73072_series_matrix.txt.gz

filename = 'data/GSE73072_series_matrix.txt'

df = pandas.read_csv(filename, skiprows=82, sep='\t', index_col='ID_REF')
df = df.iloc[:-1]

df_meta = pandas.read_csv(filename, skiprows=55, sep='\t', header=None, nrows=3)

df_meta = df_meta.T
df_meta = df_meta.iloc[1:]
hours = np.array([int(z[-1]) for z in df_meta[1].str.split(' ')])
df_meta['time'] = hours

#df_meta.columns = df_meta[0]

study = np.array([z[2] for z in df_meta[2].str.split(' ')])
mask2 = study == 'DEE2'

#

mask1 = (df_meta['time'] < 24) * (df_meta['time'] >= 0)


mask = mask1*mask2

#h1n1_only = df_meta[0]=='virus: H1N1'
idxs = np.where(mask)[0] # indices where True

pca = decomposition.PCA()

matrix = df.iloc[:, idxs].values.T

if False:

    ### 
    # pca for scatter plot only.
    X = pca.fit_transform(matrix)

    #
    hours = np.array([int(z[-1]) for z in df_meta[1].str.split(' ')[mask]])


    fig, ax = plt.subplots()

    ax.scatter(X[:,0], X[:,1], c=hours, vmin=0, vmax=120, cmap=plt.cm.magma)


