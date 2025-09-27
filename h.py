import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_config = 15625  # To match hierarchical N for comparison (5**6 = 15625; close to 10^4)
gamma = 2.5  # Power-law exponent for configuration model
k_min = 3  # Minimum degree to ensure connectivity
steps = 20  # Number of fractions for removal simulation

# Function to generate power-law degree sequence for configuration model
def generate_powerlaw_degrees(N, gamma, k_min):
    max_degree = int(N ** (1 / (gamma - 1)))  # Finite-size cutoff
    k = np.arange(k_min, max_degree + 1)
    probs = k ** (-gamma)
    probs /= probs.sum()
    degrees = np.random.choice(k, size=N, p=probs)
    if sum(degrees) % 2 != 0:
        degrees[-1] += 1
    return degrees

# Function to generate hierarchical network (Ravasz-Barabási model)
def generate_hierarchical(level):
    if level == 1:
        G = nx.complete_graph(5)
        central = 0
        external = [1, 2, 3, 4]
        return G, central, external

    G_prev, central_prev, external_prev = generate_hierarchical(level - 1)
    G = G_prev.copy()
    node_offset = len(G)
    replicas_offsets = []
    new_external = []
    for _ in range(4):
        H = G_prev.copy()
        mapping = {u: u + node_offset for u in H.nodes()}
        H = nx.relabel_nodes(H, mapping)
        G = nx.union(G, H)
        replicas_offsets.append(node_offset)
        rep_external = [e + node_offset for e in external_prev]
        new_external += rep_external
        for ex in rep_external:
            G.add_edge(ex, central_prev)
        node_offset += len(G_prev)
    new_central = central_prev
    return G, new_central, new_external

# Function to simulate targeted removal
def simulate_targeted_removal(G, metric_key, steps=20):
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []
    # Compute metrics
    if metric_key == 'degree':
        metrics = {n: d for n, d in G.degree()}
    elif metric_key == 'clustering':
        metrics = nx.clustering(G)
    # Sort nodes by metric descending
    sorted_nodes = sorted(G.nodes(), key=lambda n: metrics[n], reverse=True)
    for f in fractions:
        num_remove = int(f * N)
        remove_list = sorted_nodes[:num_remove]
        G_copy = G.copy()
        G_copy.remove_nodes_from(remove_list)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = list(nx.connected_components(G_copy))
            largest_components.append(max(len(c) for c in components) / N)
    return fractions, largest_components

# Generate configuration model network
config_degrees = generate_powerlaw_degrees(N_config, gamma, k_min)
G_config = nx.configuration_model(config_degrees, create_using=nx.Graph)
G_config = nx.Graph(G_config)  # Remove multi-edges/self-loops

# Generate hierarchical network (level 6, N=15625)
G_hier, _, _ = generate_hierarchical(6)

# Simulate attacks on configuration model
config_fracs_deg, config_sizes_deg = simulate_targeted_removal(G_config, 'degree', steps)
config_fracs_clus, config_sizes_clus = simulate_targeted_removal(G_config, 'clustering', steps)

# Simulate attacks on hierarchical network
hier_fracs_deg, hier_sizes_deg = simulate_targeted_removal(G_hier, 'degree', steps)
hier_fracs_clus, hier_sizes_clus = simulate_targeted_removal(G_hier, 'clustering', steps)

# Plot results for configuration model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(config_fracs_deg, config_sizes_deg, 'r-', label='High Degree Removal')
plt.plot(config_fracs_clus, config_sizes_clus, 'b-', label='High Clustering Removal')
plt.xlabel('Fraction Removed (f)')
plt.ylabel('Giant Component Size S(f)/N')
plt.title('Configuration Model (γ=2.5)')
plt.legend()
plt.grid(True)

# Plot results for hierarchical model
plt.subplot(1, 2, 2)
plt.plot(hier_fracs_deg, hier_sizes_deg, 'r-', label='High Degree Removal')
plt.plot(hier_fracs_clus, hier_sizes_clus, 'b-', label='High Clustering Removal')
plt.xlabel('Fraction Removed (f)')
plt.ylabel('Giant Component Size S(f)/N')
plt.title('Hierarchical Model')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Analysis (as comments):
# For both networks, high degree removal (red) reduces the giant component faster than high clustering removal (blue).
# Thus, degree is more sensitive topological information; protecting high-degree individuals limits damage best, as removing them fragments the network
