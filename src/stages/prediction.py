import h2o
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator

from src.decorators import duration
from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log

TARGET_ACTION = 9
PREDS_LIMIT = 25
SUMBIT_LINES_LIMIT = 3000
SCORE_THRESHOLD = 0.0 # not used at the moment


@duration
def filter_data_to_train(inters: pd.DataFrame, late_days: int = 3, earliest_days: int = 27) -> pd.DataFrame:
        late_date = inters['datetime'].max() - pd.Timedelta(late_days, unit='D')
        earliest_date = inters['datetime'].max() - pd.Timedelta(earliest_days, unit='D')
        pred_data = inters[
            (inters['datetime'] > earliest_date)
            &
            (inters['datetime'] < late_date)
            ]
        return pred_data

@duration
def prepare_data_to_predict(pred_data: pd.DataFrame,
                            gbm_model: H2OGradientBoostingEstimator,
                            columns_to_pred_on: list) -> pd.DataFrame:
    t = h2o.H2OFrame(pred_data[list(columns_to_pred_on)])
    t['category_id'] = t['category_id'].asfactor()
    preds = gbm_model.predict(t)

    # Обьединяем предсказания с исходными данными, чтобы фильтровать дальше
    res = pd.concat(
        [
            pred_data[list(columns_to_pred_on) + ['weight', 'user_id', 'item_id', 'datetime']].reset_index(drop=True),
            preds.as_data_frame().reset_index(drop=True)
        ],
        axis='columns',
    )
    return res


@duration
def predict_purchasers(predictions: pd.DataFrame, test_purchasers: pd.DataFrame, pred_purchasers: pd.DataFrame) -> pd.DataFrame:
    # Максимальный скор с которым предсказали целевое действие
    predictions['score'] = predictions[predictions.columns[-10:]].max(axis=1)
    # Фильтруем результаты предсказания по целевому действию 9 - покупка
    mask = (
        (predictions.predict.isin([TARGET_ACTION]) & (predictions.score > SCORE_THRESHOLD))
    )

    # Оцениваем размер таблицы после фильрации
    log.info(f'Predictions data size: {predictions[mask].shape}')

    # Оцениваем количество пользователей, которое удалось "зацепить" фильтрацией из теста
    ml_users = predictions[
        mask
        & (predictions.user_id.isin(test_purchasers.user_id.unique()))
    ].groupby('user_id')['item_id'].agg(lambda x: list(set(map(str, x)))[:PREDS_LIMIT]).reset_index().user_id.to_list()

    log.info(
        f'\nNumber of candidates by filtering: {len(pred_purchasers.user_id.unique())}\n'
        f'Number of candidates by machine learning: {len(ml_users)}\n'
        f'How many users from ml in preds by filtering: {len(set(ml_users).intersection(set(pred_purchasers.user_id.unique())))}\n'
    )

    # Генерим таблицу с кандидатами по результатам мл - предсказания
    ml_pred = (
        predictions[mask]
        .sort_values(by=['weight'], ascending=False)
        .groupby('user_id')['item_id']
        .agg(lambda x: [y for y in (set(map(str, x))) if y != '-1'][:PREDS_LIMIT])
        .reset_index()
    )
    # Напоминание о размере таблицы с кандидатами после обычной фильтрации - оставшееся от 3000 строк место мы заполнили результатами мл
    log.info(f'Merging candidates by ml of size {ml_pred.shape} with preds by {pred_purchasers.shape}')
    data_to_check = pd.concat(
        [
            ml_pred.reset_index(drop=True),
            pred_purchasers.reset_index(drop=True),
        ],
        axis='rows'
    )[-SUMBIT_LINES_LIMIT:]

    data_to_check = data_to_check.drop_duplicates(subset='user_id', keep='last')
    log.info(f'\nMerging resulted in data of size: {data_to_check.shape}\n'
             f'Number of predicted users found in test data: {data_to_check[data_to_check.user_id.isin(test_purchasers.user_id.unique())].shape}')
    return data_to_check
