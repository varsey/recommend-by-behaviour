import pandas as pd

from src.decorators import duration
from src.stages.helpers import calculate_recall
from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log


@duration
def estimate_recall(test_purchasers: pd.DataFrame, pred_purchasers: pd.DataFrame, threshold: int = 3000) -> None:
    check_merge = test_purchasers.merge(pred_purchasers[-threshold:], on='user_id', how='left')
    check_merge.item_id_y.loc[check_merge.item_id_y.isnull()] = check_merge.item_id_y.loc[check_merge.item_id_y.isnull()].apply(lambda x: [])
    check_merge['recall'] = check_merge.apply(calculate_recall, axis=1)
    log.info(
        f'Recall on this data: {check_merge["recall"].mean().round(4)}'
    )
