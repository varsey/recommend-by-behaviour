def processing(purchasers):
    purchasers = purchasers.map(lambda x: [f'{y}' for y in x] if isinstance(x, list) else [f'{x}'])
    purchasers = list([item for sublist in purchasers for item in sublist])
    return list(set(purchasers))[:25]

def calculate_recall(row):
    set_col1 = set(row['item_id_x'])
    set_col2 = set(row['item_id_y'])

    true_positives = len(set_col1.intersection(set_col2))
    false_negatives = len(set_col1) # len(set_col2 - set_col1)

    if true_positives + false_negatives == 0:  # Avoid division by zero
        return 0.0
    return true_positives / (true_positives + false_negatives)