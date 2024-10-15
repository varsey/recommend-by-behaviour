import pandas as pd

from src.decorators import duration
from src.stages.helpers import processing
from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log

KEY_ACTION_CODE = 9 # buy is action 9, see etl for details
ACTION_7_LIMIT = 50
ACTION_8_LIMIT = 20
ACTION_9_LIMIT = 0
ACTIONS_TO_FILTER = [8, 1]

@duration
def get_test_data(inters: pd.DataFrame, start_days: int = 3, end_days: int = 0) -> pd.DataFrame:
    # Тестовые данные - три дня и ключевое событие - 9 (заказ)
    start_date = inters['datetime'].max() - pd.Timedelta(start_days, unit='D')
    end_date = inters['datetime'].max() - pd.Timedelta(end_days, unit='D')
    test_purchasers = inters.sort_values(by=['datetime', 'weight'], ascending=True)
    test_purchasers = test_purchasers[
        (test_purchasers['weight'] == KEY_ACTION_CODE)
        & (test_purchasers['datetime'] > start_date)
        & (test_purchasers['datetime'] < end_date)
    ]
    print(test_purchasers['datetime'].max(), test_purchasers['datetime'].min(),)
    test_purchasers = test_purchasers.groupby(['user_id'])['item_id'].apply(processing)
    test_purchasers = test_purchasers.reset_index()
    log.info(f'Test data has shape: {test_purchasers.shape}')
    return test_purchasers

@duration
def get_core_candidates(inters: pd.DataFrame, days_to_shift: int =3) -> pd.DataFrame:
    last_day = inters['datetime'].max() - pd.Timedelta(
        days_to_shift,
        unit='D'
    )
    first_day = inters['datetime'].max() - pd.Timedelta(
        (5 + days_to_shift) * 24,
        unit='hours'
    )
    pred_purchasers = inters.sort_values(by=['datetime', 'weight'], ascending=True)
    pred_purchasers = pred_purchasers[
        (pred_purchasers['weight'].isin(ACTIONS_TO_FILTER))
        &
        (pred_purchasers['datetime'] > first_day)
        &
        (pred_purchasers['datetime'] < last_day)
        &
        (
            (pred_purchasers.action_7_count > ACTION_7_LIMIT)
            |
            (pred_purchasers.action_8_count > ACTION_8_LIMIT)
            |
            (pred_purchasers.action_9_count > ACTION_9_LIMIT)
        )
    ]
    pred_purchasers = pred_purchasers.groupby(['user_id'])['item_id'].apply(processing)
    pred_purchasers = pred_purchasers.reset_index()
    log.info(f'Test data has shape: {pred_purchasers.shape}')
    return pred_purchasers
