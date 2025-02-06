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
    import numpy as np
    import matplotlib.pyplot as plt
 
    x = np.linspace(-5,5,10)

    # x = np.reshape(x, (len(x),1)) # for handling 1D only.
    # p = np.exp(-(x-3)**2/25)
    # p = p/np.sum(p)
    # q = np.exp(-(x+3)**2/25)
    # q = q/np.sum(q)

    x,y = np.meshgrid(x, x) # 2d

    a,b = np.shape(x)  
    x = np.reshape(x, (b**2,))
    y = np.reshape(y, (b**2,))
    

    
    X = np.vstack([x,y]).T



    # y = np.reshape(x[:,0], (len(x[:,0]),1))
    #p = np.exp(-(y-3)**2- (x-2)**2 )
    #q = np.exp(-(y+3)**2 - (x+2)**2)
    #q = q/np.sum(q)
    #p = p/np.sum(p)

    # p = np.zeros((b**2,1)) # one point
    # q = np.zeros((b**2,1))
    # p[10] = 1
    # q[5] = 1

    p = np.zeros((b**2,1))
    p[int(b**2/2):-1] = 1
    p[-1] = 2/b**2
    q = np.zeros((b**2,1))
    q[0:int(b**2/2)] = 1
    p = p/p.sum()
    q = q/q.sum()
 
    cost, T = demo_wasserstein(X,p,q)
    
    # add jitter (TODO: why does this break code when done before demo.wasserstein?)
    # ONLY for visualization for now.
    #x = x + 0.1*np.random.randn(b**2)
    #y = y + 0.1*np.random.randn(b**2)
    
    plt.scatter(x,y, c='blue', alpha=p/max(p))
    plt.scatter(x,y, c='red', alpha=q/max(q))
    #x = np.reshape( x, (a,b,c))
    #for i in range(b):
    #    for j in range(b):
    #        r=np.ndarray.item(p[i])
    #        plt.plot(x[0,i,j], x[0,i,j], color=(.9,0,0), marker=".")
    # for i in range(len(q)):
    #     r=np.ndarray.item(p[i])
    #     plt.plot(x[i,0], x[i,1], color=(.9,0,0), marker=".")
    plt.show(block=False)
    
    # goal: draw little lines between every pair of points that 
    # mass got moved between.
    # T_ij is such that moving mass T[i,j] from position (x[i],y[i]) to (x[j],y[j])
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if abs(T[i,j]) > 1e-3: #idk
                _x = [ x[i], x[j] ]
                _y = [ y[i], y[j] ]
                plt.plot(_x, _y, lw=10*T[i,j],  c='k')
    plt.show(block=False)
