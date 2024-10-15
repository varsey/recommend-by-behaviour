import pandas as pd
from rectools.models import ImplicitALSWrapperModel
from rectools.dataset import Dataset

from src.logger import DicLogger, LOGGING_CONFIG
from src.stages.prediction import PREDS_LIMIT

log = DicLogger(LOGGING_CONFIG).log


def enrich_preds_with_recos(model: ImplicitALSWrapperModel, predicts: pd.DataFrame, recs_dataset: Dataset):
    recos = model.recommend(
        users=predicts.user_id,
        dataset=recs_dataset,
        k=22,
        filter_viewed=False,
    )

    recos = recos[recos.item_id > -1]
    # Дополняем данные с предыдущих предсказаний (фильтрация + ml)

    recos_agg = recos.groupby('user_id')['item_id'].agg(lambda x: list(map(str, x))).reset_index()

    recos_agg_selected_uid = recos_agg[
        recos_agg.user_id.isin(
            predicts.user_id
        )
    ]

    # Объединяем рекомендации с предсказанями фильтрацией и мл
    enriched_data_to_check = predicts.merge(recos_agg_selected_uid, on='user_id', how='left')
    enriched_data_to_check['products_count'] = enriched_data_to_check['item_id_x'].apply(len)
    log.info(f'Mean number of items per recommendation before recos added: '
             f'{enriched_data_to_check["products_count"].mean().round(4)}')
    enriched_data_to_check['products_count'].mean()
    # Заполняем nan пустыми списками для корректного расчета метрики
    enriched_data_to_check.item_id_y.loc[enriched_data_to_check.item_id_y.isnull()] = enriched_data_to_check.item_id_y.loc[enriched_data_to_check.item_id_y.isnull()].apply(lambda x: [])
    # Обьединяем исходное предсказание рекомендациями по rectools - сначала исходные айди, потом рекоммендации (порядок важен, тк далее оставляем только 25 элементов списка)
    enriched_data_to_check['item_id'] = enriched_data_to_check['item_id_x'] + enriched_data_to_check['item_id_y']
    # Убеждаемся что добавлением новых товаров мы не вышли
    enriched_data_to_check['item_id'] = enriched_data_to_check['item_id'].apply(lambda x: list(set(x[:PREDS_LIMIT])))

    enriched_data_to_check['products_count'] = enriched_data_to_check['item_id'].apply(len)
    log.info(f'Mean number of items per recommendation after recos added:'
             f' {enriched_data_to_check["products_count"].mean().round(4)}')

    enriched_data_to_check = enriched_data_to_check[['user_id', 'item_id']]
    log.info(f'Resut dataframe size : {enriched_data_to_check.drop_duplicates(subset=["user_id"]).shape}')

    return enriched_data_to_check
