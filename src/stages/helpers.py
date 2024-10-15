import faiss


def processing(purchasers):
    purchasers = purchasers.map(lambda x: [f'{y}' for y in x] if isinstance(x, list) else [f'{x}'])
    purchasers = list([item for sublist in purchasers for item in sublist if item != '-1'])
    return list(set(purchasers))[:25]


def calculate_recall(row):
    set_col1 = set(row['item_id_x'])
    set_col2 = set(row['item_id_y'])

    true_positives = len(set_col1.intersection(set_col2))
    false_negatives = len(set_col1) # len(set_col2 - set_col1)

    if true_positives + false_negatives == 0:  # Avoid division by zero
        return 0.0
    return true_positives / (true_positives + false_negatives)


def create_index_hnsw(tfidf_matrix):
    index_hnsw = faiss.IndexHNSWFlat(tfidf_matrix.shape[1], 32)
    index_hnsw.train(tfidf_matrix)
    index_hnsw.add(tfidf_matrix)
    return index_hnsw


def get_similar(x, catalog, index, vectorizer):
    res = []
    for y in x[:4]:
        cand = catalog[catalog.product_id == y]
        if cand.shape[0] > 0:
            to_match = cand.desc.values[0]
            query_vector = vectorizer.transform([to_match, ])
            _, indices = index.search(query_vector.toarray().reshape(1, -1), 2)
            match1 = catalog.iloc[indices[0][0]].to_dict()
            match2 = catalog.iloc[indices[0][1]].to_dict()
            res.append(match1['product_id'])
            res.append(match2['product_id'])
    return res
