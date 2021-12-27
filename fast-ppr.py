import numpy as np
from scipy import sparse
from fast_pagerank import pagerank, pagerank_power

A = np.array([[0,1], [0, 2], [1, 2], [2, 0], [3, 2]])
weights = [0.7, 0.5, 0.6, 0.9, 0.8]
print(A[:,0])
G = sparse.csr_matrix((weights, (A[:,0], A[:,1])), shape=(4, 4))
print(G[1, 1])
pr = pagerank(G, p=0.85)
print(pr)

personalize = np.array([0.4, 0.2, 0.2, 0.4])
pr = pagerank_power(G, p=0.85, personalize=personalize, tol=1e-6)
print(pr)