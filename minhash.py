import numpy as np
import math
import random

# ------------------ MinHash Part ------------------

def build_shingle_matrix(docs, k=2):
    """
    Build binary shingle-document matrix using k-word shingles.
    """
    doc_shingles = [{" ".join(doc.lower().split()[i:i+k]) for i in range(len(doc.split())-k+1)} for doc in docs]
    shingles = list(set().union(*doc_shingles))
    shingle_index = {s: i for i, s in enumerate(shingles)}

    M = np.zeros((len(shingles), len(docs)), dtype=int)
    for j, s_set in enumerate(doc_shingles):
        for s in s_set:
            M[shingle_index[s], j] = 1
    return M, shingles

def minhash_signature(M, hash_functions):
    """Compute MinHash signature matrix using given hash functions."""
    n_shingles, n_docs = M.shape
    num_hashes = len(hash_functions)
    sig = np.full((num_hashes, n_docs), np.inf)

    for r in range(n_shingles):
        hash_values = [h(r) for h in hash_functions]
        for c in range(n_docs):
            if M[r, c] == 1:
                sig[:, c] = np.minimum(sig[:, c], hash_values)
    return sig
"""
def minhash_signature(M, num_hashes=100):
    n_shingles, n_docs = M.shape
    sig = np.full((num_hashes, n_docs), np.inf)
    p = 2**31 - 1
    hash_funcs = [(random.randint(1, p-1), random.randint(0, p-1)) for _ in range(num_hashes)]

    for r in range(n_shingles):
        for h, (a, b) in enumerate(hash_funcs):
            hash_val = (a*r + b) % p
            sig[h, M[r] == 1] = np.minimum(sig[h, M[r] == 1], hash_val)
    return sig
"""
def minhash_signature_permutation(M, num_permutations=100):
    n_shingles, n_docs = M.shape
    sig = np.full((num_permutations, n_docs), np.inf)
    for p in range(num_permutations):
        perm = np.random.permutation(n_shingles)
        for c in range(n_docs):
            idx = next((idx for idx in perm if M[idx, c]), None)
            if idx is not None:
                sig[p, c] = idx
    return sig

def jaccard_from_signatures(sig, i, j):
    return np.mean(sig[:, i] == sig[:, j])


# ------------------ TF-IDF Part ------------------

def build_count_matrix(docs):
    vocab = {}
    tokenized_docs = [doc.lower().split() for doc in docs]
    for tokens in tokenized_docs:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    count_matrix = np.zeros((len(vocab), len(docs)), dtype=float)
    for j, tokens in enumerate(tokenized_docs):
        for t in tokens:
            count_matrix[vocab[t], j] += 1
    return count_matrix, vocab

def compute_tfidf(count_matrix):
    df = np.count_nonzero(count_matrix > 0, axis=1)
    tf = count_matrix / (count_matrix.sum(axis=0) + 1e-10)
    idf = np.log((count_matrix.shape[1] + 1) / (df + 1)) + 1
    return tf * idf[:, None]

def cosine_similarity(A):
    norms = np.linalg.norm(A, axis=0)
    return (A.T @ A) / (norms[:, None] * norms[None, :] + 1e-10)

def euclidean_distance(A):
    return np.sqrt(((A[:, :, None] - A[:, None, :])**2).sum(axis=0))


# ------------------ Example Usage ------------------

if __name__ == "__main__":
    docs = [
        "the quick brown fox",
        "the quick brown dog",
        "the fast brown fox"
    ]

    print("\n=== MinHash (Binary Shingle Matrix) ===")
    M, shingles = build_shingle_matrix(docs, k=2)
    print(M, shingles)
    modulus = 5
    h1 = lambda r: (1*r + 1) % modulus
    h2 = lambda r: (3*r + 1) % modulus
    hash_functions = [h1, h2]

    # Compute signature matrix
    sig = minhash_signature(M, hash_functions)
    print(sig)
    #sig = minhash_signature(M, num_hashes=50)
    sig_perm = minhash_signature_permutation(M, num_permutations=50)
    print(sig_perm)
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            print(f"MinHash (Hash): Doc {i}-{j}:", jaccard_from_signatures(sig, i, j))
            print(f"MinHash (Permutation): Doc {i}-{j}:", jaccard_from_signatures(sig_perm, i, j))

    print("\n=== TF-IDF (Cosine & Euclidean) ===")
    count_matrix, vocab = build_count_matrix(docs)
    tfidf = compute_tfidf(count_matrix)
    print("Cosine Similarity:\n", cosine_similarity(tfidf))
    print("Euclidean Distance:\n", euclidean_distance(tfidf))
