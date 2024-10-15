import numpy as np
import pandas as pd

from src.decorators import duration


@duration
def load_data(catalog_path: str, actions_path: str) -> (pd.DataFrame, pd.DataFrame):
    catalog = pd.read_parquet(catalog_path, engine='pyarrow')
    actions = pd.read_parquet(actions_path, engine='pyarrow')

    # Фильтруем ложные айдишники
    catalog = catalog[catalog['product_id'] != '/*']
    actions = actions[actions['user_id'] != '/*']

    _actions = (
        actions
        .copy()
        .explode('products')
        .assign(products=lambda x: x.products.fillna(-1).astype(np.int64))
    )

    _catalog = (
        catalog
        .copy()
        .assign(product_id=lambda x: x.product_id.astype(np.int64))
    )

    # Разворачиваем данные по айдишникам продуктов
    _actions['action_datetime'] = pd.to_datetime(_actions['date'])

    # Новая шкала action в train_actions.pq в порядке возрастания важности
    _actions.action = _actions.action + 10
    _actions.action = _actions.action.replace({14: 0, 13: 4, 18: 2, 19: 3, 10: 5, 17: 1, 11: 6, 12: 7, 16: 8, 15: 9})
    _actions = _actions.sort_values(by=['action', 'date'])

    #     0 - clearB (удаление всех товаров из корзины)
    #     1 - visit (посещение страницы с товаром)
    #     2 - visitCategory (посещение страницы с группой товаров)
    #     3 - search (поиск товара)
    #     4 - delB (удаление товара из корзины)
    #     5 - view (просмотр товара)
    #     6 - like (лайк товара)
    #     7 - addB (добавление товара в корзину)
    #     8 - listB (посещение страницы корзины и вывод списка товаров в корзине)
    #     9 - order (оформление заказа)

    # Переименовываем поля и обьединияем датасеты каталога и дейсвтвий пользователя
    # 'products' теперь 'item_id', 'action' превратился в 'weight' (мы перевели код действия в шкалу по возрастанию ранее)
    interactions = _actions.rename(columns={'products': 'item_id', 'action' : 'weight', 'action_datetime': 'datetime'})
    items = _catalog.rename(columns={'product_id': 'item_id'})

    return interactions, items, _catalog.drop_duplicates(subset='product_id')

@duration
def generate_features(interactions: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    interactions_merge = interactions.merge(items, on='item_id', how='left')

    # Генерим новые фичи
    interactions_merge['day'] = interactions_merge['datetime'].dt.day
    interactions_merge['day_of_week'] = interactions_merge['datetime'].dt.dayofweek

    interactions_merge['hour'] = interactions_merge['datetime'].dt.hour
    interactions_merge['minute'] = interactions_merge['datetime'].dt.minute

    interactions_merge['price_diff'] = interactions_merge['price'] - interactions_merge['old_price']

    interactions_merge['category_id'] = interactions_merge['category_id'].fillna(0)
    interactions_merge['price_diff'] = interactions_merge['price_diff'].fillna(0)

    interactions_merge['category_id'] = interactions_merge['category_id'].astype(np.int32)

    inters = interactions_merge.merge(
        interactions_merge.groupby('user_id')['loc_user_id'].count().reset_index(name='loc_user_count'),
        on='user_id',
        how='left'
    )

    inters = inters.merge(
        interactions_merge.groupby('user_id')['category_id'].count().reset_index(name='category_id_count'),
        on='user_id',
        how='left'
    )

    for action in sorted(interactions_merge.weight.unique()):
        inters = inters.merge(
            interactions_merge[interactions_merge.weight == action].groupby('user_id')['weight'].count().reset_index(
                name=f'action_{action}_count'),
            on='user_id',
            how='left'
        )
        inters[f'action_{action}_count'] = inters[f'action_{action}_count'].fillna(0)

    inters['delta_sec'] = inters.sort_values(by=['datetime']).groupby('user_id')['datetime'].diff()
    inters['delta_sec'] = inters['delta_sec'].dt.total_seconds()
    inters['delta_sec'] = inters['delta_sec'].fillna(0)

    inters['delta_day'] = inters.sort_values(by=['datetime']).groupby('user_id')['day'].diff()
    inters['delta_day'] = inters['delta_day'].fillna(0)

    inters['delta_hour'] = inters.sort_values(by=['datetime']).groupby('user_id')['hour'].diff()
    inters['delta_hour'] = inters['delta_hour'].fillna(0)

    inters['delta_min'] = inters.sort_values(by=['datetime']).groupby('user_id')['minute'].diff()
    inters['delta_min'] = inters['delta_min'].fillna(0)

    inters = inters.drop(columns=['day', 'hour', 'minute'])

    return inters