import numpy as np
import pandas as pd
from numpy.linalg import svd, norm

def build_count_matrix(docs):
    vocab = {}
    tokenized_docs = [doc.lower().split() for doc in docs] #tokenized_docs = [[w for w in doc.lower().split() if w not in STOPWORDS]for doc in docs]
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
    tf = count_matrix / (count_matrix.sum(axis=0) + 1e-10)     # term frequency
    idf = np.log((count_matrix.shape[1] + 1) / (df + 1)) + 1   # smooth IDF
    return tf * idf[:, None]

def rank_docs(query, vocab, count_matrix, U, S, Vh, sentences):
    # Query count vector
    q_vec = np.zeros(len(vocab))
    for w in query.lower().split():
        if w in vocab:#if w not in STOPWORDS and w in vocab:
            q_vec[vocab[w]] += 1

    # TF-IDF for query
    df = np.count_nonzero(count_matrix > 0, axis=1)
    tf = q_vec / (q_vec.sum() + 1e-10)
    idf = np.log((count_matrix.shape[1] + 1) / (df + 1)) + 1
    q_tfidf = tf * idf

    # Project query and docs
    q_lsa = (q_tfidf @ U) @ S
    docs_lsa = (S @ Vh).T

    # Cosine similarity
    sims = docs_lsa @ q_lsa / (norm(docs_lsa, axis=1) * norm(q_lsa) + 1e-10)
    order = np.argsort(sims)[::-1]
    return [(sentences[i], sims[i]) for i in order]

if __name__ == "__main__":
    sentences = [
        "Data science is fun",
        "Machine learning is part of data science",
        "Deep learning is a subset of machine learning"
    ]

    # Build count matrix & vocab
    count_matrix, vocab = build_count_matrix(sentences)

    # TF-IDF
    tfidf = compute_tfidf(count_matrix)
    print("TF-IDF Matrix:\n", pd.DataFrame(np.round(tfidf, 3), index=vocab))

    # SVD -> reduce to 2 dimensions
    U, S_full, Vh = svd(tfidf)
    k = 2
    U, S, Vh = U[:, :k], np.diag(S_full[:k]), Vh[:k, :]

    # Query
    query = "machine"
    ranked = rank_docs(query, vocab, count_matrix, U, S, Vh, sentences)

    print(f"\nRanking for query '{query}':")
    for doc, sc in ranked:
        print(f"{sc:.4f} => {doc}")
