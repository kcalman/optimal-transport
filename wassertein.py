from scipy import optimize
from sklearn import metrics


def demo_wasserstein(x, p, q):
    
    """
    Computes order-2 Wasserstein distance between two
    discrete distributions.
 
    Parameters
    ----------
    x : ndarray, has shape (num_bins, dimension)
 
        Locations of discrete atoms (or "spatial bins")
 
    p : ndarray, has shape (num_bins,)
 
        Probability mass of the first distribution on each atom.
 
    q : ndarray, has shape (num_bins,)
 
        Probability mass of the second distribution on each atom.
 
    Returns
    -------
    dist : float
 
        The Wasserstein distance between the two distributions.
 
    T : ndarray, has shape (num_bins, num_bins)
 
        Optimal transport plan. Satisfies p == T.sum(axis=0)
        and q == T.sum(axis=1).
 
    Note
    ----
    This function is meant for demo purposes only and is not
    optimized for speed. It should still work reasonably well
    for moderately sized problems.
    """
    
    # Check inputs.
    if (abs(p.sum() - 1) > 1e-9) or (abs(p.sum() - q.sum()) > 1e-9):
        raise ValueError("Expected normalized probability masses.")
 
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Expected nonnegative mass vectors.")
 
    if (x.shape[0] != p.size) or (p.size != q.size):
        raise ValueError("Dimension mismatch.")
 
    # Compute pairwise costs between all xs.
    n, d = x.shape
    #C = squareform(pdist(x, metric="sqeuclidean"))
    C = metrics.pairwise_distances(x)
 
    # Scipy's linear programming solver will accept the problem in
    # the following form:
    # 
    # minimize     c @ t        over t
    # subject to   A @ t == b
    #
    # where we specify the vectors c, b and the matrix A as parameters.
 
    # Construct matrices Ap and Aq encoding marginal constraints.
    # We want (Ap @ t == p) and (Aq @ t == q).
    Ap, Aq = [], []
    z = np.zeros((n, n))
    z[:, 0] = 1
 
    for i in range(n):
        Ap.append(z.ravel())
        Aq.append(z.transpose().ravel())
        z = np.roll(z, 1, axis=1)
 
    # We can leave off the final constraint, as it is redundant.
    # See Remark 3.1 in Peyre & Cuturi (2019).
    A = np.row_stack((Ap, Aq))[:-1]
    b = np.concatenate((p, q))[:-1]
 
    # Solve linear program, recover optimal vector t.
    result = optimize.linprog(C.ravel(), A_eq=A, b_eq=b)
 
    # Reshape optimal vector into (n x n) transport plan matrix T.
    T = result.x.reshape((n, n))
 
    # Return Wasserstein distance and transport plan.
    return np.sqrt(np.sum(T * C)), T
#
 
if __name__=="__main__":
    import pandas
    from matplotlib import pyplot as plt
    from sklearn import decomposition
    from wassertein import wasserstein
    import numpy as np


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

    # h1n1_only = df_meta[0]=='virus: H1N1'
    idxs = np.where(mask)[0] # indices where True

    matrix = df.iloc[:, idxs].values.T
 


    x = np.linspace(0,12023,12023)
    x = np.reshape(x, (len(x),1)) # for handling 1D only.
    p = matrix[0,:]
    p = p/p.sum()
    q = matrix[1,:]
    q = q/q.sum()

 
    cost, T = demo_wasserstein(x,p,q)



