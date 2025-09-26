import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def pagerank(G, damping=0.85, max_iterations = 100, tolerance = 1e-6):
    nodes = sorted(G.nodes())
    n = len(nodes)
    adj_matrix = nx.to_numpy_array(G,nodelist=nodes,dtype=float)
    print("Adjacency Matrix",adj_matrix)
    
    H = np.zeros((n,n))
    for i in range(n):
        rowsum = np.sum(adj_matrix[i])
        if rowsum == 0:
            H[i] = np.ones(n)/n
        else:
            H[i] = adj_matrix[i]/rowsum
            
    print("Hyper link matrix", H)
    
    G_matrix = damping * H + (1- damping) * (np.ones((n,n))/n)
    print("G matrix",G_matrix)
    
    #WITH POWER_ITERATION
    """PR = np.ones(n)/n
    print("Initial PR", PR)
    
    for iteration in range(max_iterations):
        new_PR = PR @ G_matrix
        print(f"PR({iteration+1}) : {new_PR}")
        
        if np.allclose(PR, new_PR,atol=tolerance):
            print(f"Coverged after {iteration+1} iterations")
            break
        
        PR = new_PR"""
    #WITH_EIGENVALUES
    eigenvals, eigenvec = np.linalg.eig(G_matrix.T)
    idx = np.argmin(np.abs(eigenvals - 1))
    principal_eigenvec = np.real(eigenvec[:,idx])
    PR = principal_eigenvec/np.sum(principal_eigenvec)
        
    result = {nodes[i] : PR[i] for i in range(n)}
    print("Final Page rank",result)
    
    return result
    
def visualize(G, scores):
  nx.draw(
      G,
      with_labels=True,
      node_size=[scores[n]*3000 for n in G.nodes()],
      arrows = True
  )
  plt.show()
    
G = nx.DiGraph()
edges = [
    ('B', 'D'),
    ('C', 'A'), ('C', 'B'),
    ('D', 'C')
]

G.add_edges_from(edges)

scores = pagerank(G)
visualize(G, scores)