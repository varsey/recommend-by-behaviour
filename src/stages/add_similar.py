import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import create_index_hnsw
from src.decorators import duration
from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log



def add_similar_items(catalog: pd.DataFrame, enriched_data: pd.DataFrame) -> pd.DataFrame:
    catalog['desc'] = catalog['category_id'].astype(str) + ' ' + catalog['title']
    catalog = catalog.drop_duplicates(subset='product_id').reset_index(drop=True)
    catalog = catalog.dropna(subset=['desc'])

    documents = catalog.desc.dropna().to_list()
    print(len(documents))
    vectorizer = TfidfVectorizer(lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(documents).toarray().astype('float32')

    index = create_index_hnsw(tfidf_matrix)

    def get_similar(x):
        res = x
        for y in x:
            cand = catalog[catalog.product_id == y]
            if cand.shape[0] > 0:
                to_match = cand.desc.values[0]
                query_vector = vectorizer.transform([to_match, ])
                _, indices = index.search(query_vector.toarray().reshape(1, -1), 2)
                match1 = catalog.iloc[indices[0][0]].to_dict()
                match2 = catalog.iloc[indices[0][1]].to_dict()
                if match1['product_id'] not in res and len(res) < 25:
                    res.append(match1['product_id'])
                if match2['product_id'] not in res and len(res) < 25:
                    res.append(match2['product_id'])
            if len(res) >= 25:
                return res
        return res

    enriched_data_to_check_w_similar = enriched_data.copy()

    enriched_data_to_check_w_similar['products_count'] = enriched_data_to_check_w_similar['item_id'].apply(len)
    enriched_data_to_check_w_similar['products_count'].mean().round(4)

    enriched_data_to_check_w_similar['items_sim'] = enriched_data_to_check_w_similar['item_id'].apply(
        lambda x: get_similar(x))

    # Убеждаемся что добавлением новых товаров мы не вышли
    enriched_data_to_check_w_similar['item_id'] = enriched_data_to_check_w_similar['item_id'] + \
                                                  enriched_data_to_check_w_similar['items_sim'].apply(
                                                      lambda x: list(set(x)))

    enriched_data_to_check_w_similar['item_id'] = enriched_data_to_check_w_similar['item_id'].apply(
        lambda x: list(set(x[:25])))

    enriched_data_to_check_w_similar['products_count'] = enriched_data_to_check_w_similar['item_id'].apply(len)
    enriched_data_to_check_w_similar['products_count'].mean().round(4)

    return enriched_data_to_check_w_similar