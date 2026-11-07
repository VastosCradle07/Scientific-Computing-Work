import numpy as np

pages = ["Home", "Blog", "Shop", "About", "Contact"]

links = {
    0: [1, 2],
    1: [2],
    2: [0],
    3: [2, 4],
    4: [3]
}

n = len(pages)
damping = 0.85

M = np.zeros((n, n))
for source, targets in links.items():
    for target in targets:
        M[target, source] = 1.0 / len(targets)

G = damping * M + (1 - damping) / n * np.ones((n, n))

eigenvalues, eigenvectors = np.linalg.eig(G)

index = np.argmin(np.abs(eigenvalues - 1.0))
vector = np.real(eigenvectors[:, index])
rank = vector / vector.sum()

power_rank = np.full(n, 1.0 / n)
for _ in range(200):
    power_rank = G @ power_rank

print("Dominant eigenvalue:", np.real(eigenvalues[index]))
print("Column sums of G:", G.sum(axis=0))
print("PageRank scores from eigenvector:")
for name, score in sorted(zip(pages, rank), key=lambda item: -item[1]):
    print(f"{name}: {score:.4f}")
print("Power iteration agrees:", np.allclose(rank, power_rank))
print("Verification error:", np.linalg.norm(G @ rank - rank))
