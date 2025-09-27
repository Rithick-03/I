#Running for Alot of time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000  # Number of nodes
gamma = 2.2  # Power-law exponent
k_min = 2  # Minimum degree
rewiring_steps = 100000  # Number of rewiring steps for XB&S algorithm

# Function to generate power-law degree sequence
def generate_powerlaw_degrees(N, gamma, k_min):
    """Generate degree sequence from P(k) ~ k^-gamma."""
    max_degree = int(N**(1/(gamma-1)))  # Finite-size cutoff ~ N^(1/(gamma-1))
    k = np.arange(k_min, max_degree + 1)
    probs = k**(-gamma)
    probs /= probs.sum()
    degrees = np.random.choice(k, size=N, p=probs)
    if sum(degrees) % 2 != 0:  # Ensure sum is even
        degrees[-1] += 1
    return degrees

# Xulvi-Brunet & Sokolov algorithm for assortative/disassortative rewiring
def xbs_rewiring(G, assortative=True, steps=100000):
    """Rewire network to increase (assortative=True) or decrease (assortative=False) assortativity."""
    G_copy = G.copy()
    for _ in range(steps):
        # Select two random edges
        edges = list(G_copy.edges())
        if len(edges) < 2:
            break
        e1, e2 = np.random.choice(len(edges), size=2, replace=False)
        u, v = edges[e1]
        x, y = edges[e2]
        # Ensure distinct nodes
        if u in (x, y) or v in (x, y):
            continue
        # Degrees
        du, dv, dx, dy = G_copy.degree(u), G_copy.degree(v), G_copy.degree(x), G_copy.degree(y)
        if assortative:
            # Swap to increase assortativity: connect similar-degree pairs
            if abs(du - dy) + abs(dv - dx) < abs(du - dv) + abs(dx - dy):
                G_copy.remove_edges_from([(u, v), (x, y)])
                G_copy.add_edges_from([(u, y), (x, v)])
        else:
            # Swap to decrease assortativity: connect dissimilar-degree pairs
            if abs(du - dx) + abs(dv - dy) < abs(du - dv) + abs(dx - dy):
                G_copy.remove_edges_from([(u, v), (x, y)])
                G_copy.add_edges_from([(u, x), (v, y)])
    return G_copy

# Simulate random node removal
def simulate_percolation(G, steps=20):
    """Simulate random node removal and return f, P_infty(f)/P_infty(0)."""
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []
    for f in fractions:
        G_copy = G.copy()
        nodes_to_remove = np.random.choice(list(G_copy.nodes()), size=int(f * N), replace=False)
        G_copy.remove_nodes_from(nodes_to_remove)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = list(nx.connected_components(G_copy))
            largest_components.append(max(len(c) for c in components) / N)
    return fractions, largest_components

# Generate degree sequence
degrees = generate_powerlaw_degrees(N, gamma, k_min)

# Create networks
# 1. Neutral (configuration model)
G_neutral = nx.configuration_model(degrees, create_using=nx.Graph)
G_neutral = nx.Graph(G_neutral)  # Remove multi-edges/self-loops

# 2. Assortative
G_assortative = xbs_rewiring(G_neutral, assortative=True, steps=rewiring_steps)

# 3. Disassortative
G_disassortative = xbs_rewiring(G_neutral, assortative=False, steps=rewiring_steps)

# Compute assortativity coefficients
r_neutral = nx.degree_assortativity_coefficient(G_neutral)
r_assortative = nx.degree_assortativity_coefficient(G_assortative)
r_disassortative = nx.degree_assortativity_coefficient(G_disassortative)

# Simulate percolation for each network
fracs_neutral, sizes_neutral = simulate_percolation(G_neutral)
fracs_assort, sizes_assort = simulate_percolation(G_assortative)
fracs_disassort, sizes_disassort = simulate_percolation(G_disassortative)

# Print assortativity
print("=== Assortativity Coefficients ===")
print(f"Neutral: r = {r_neutral:.3f}")
print(f"Assortative: r = {r_assortative:.3f}")
print(f"Disassortative: r = {r_disassortative:.3f}")

# Plot percolation curves
plt.figure(figsize=(8, 6))
plt.plot(fracs_neutral, sizes_neutral, 'b-', label=f'Neutral (r={r_neutral:.2f})')
plt.plot(fracs_assort, sizes_assort, 'g-', label=f'Assortative (r={r_assortative:.2f})')
plt.plot(fracs_disassort, sizes_disassort, 'r-', label=f'Disassortative (r={r_disassortative:.2f})')
plt.xlabel('Fraction of Nodes Removed (f)')
plt.ylabel('P_infty(f) / P_infty(0)')
plt.title('Robustness of Correlated Power-Law Networks (γ=2.2)')
plt.legend()
plt.grid(True)
plt.show()

# Determine critical thresholds empirically (where P_infty drops below 0.01)
def find_empirical_fc(fractions, sizes, threshold=0.01):
    for i, size in enumerate(sizes):
        if size <= threshold:
            return fractions[i]
    return 1.0

fc_neutral = find_empirical_fc(fracs_neutral, sizes_neutral)
fc_assort = find_empirical_fc(fracs_assort, sizes_assort)
fc_disassort = find_empirical_fc(fracs_disassort, sizes_disassort)

print("\n=== Empirical Critical Thresholds ===")
print(f"Neutral: fc = {fc_neutral:.3f}")
print(f"Assortative: fc = {fc_assort:.3f}")
print(f"Disassortative: fc = {fc_disassort:.3f}")
