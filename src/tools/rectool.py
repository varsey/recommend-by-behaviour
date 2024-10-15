import os
import threadpoolctl
import warnings

import pandas as pd
from rectools.models import ImplicitALSWrapperModel
from rectools.dataset import Dataset
from implicit.als import AlternatingLeastSquares

warnings.filterwarnings('ignore')

# For implicit ALS
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas")

from src.decorators import duration
from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log

K_RECOS = 25
NUM_THREADS = 24
RANDOM_STATE = 32
ITERATIONS = 25


class RecTool:
    def __init__(self):
        self.rec_model = None

    @staticmethod
    def make_base_model(factors: int, regularization: float, alpha: float, fit_features_together: bool=False):
        return ImplicitALSWrapperModel(
            AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                alpha=alpha,
                random_state=RANDOM_STATE,
                use_gpu=False,
                num_threads = NUM_THREADS,
                iterations=ITERATIONS),
            fit_features_together = fit_features_together,
            )

    @duration
    def fit_recommends(self, catalog: pd.DataFrame, recos_data: pd.DataFrame) -> Dataset:
        log.info('Start fitting recommendation model')
        items = catalog.rename(columns={'product_id': 'item_id'})

        items = items.loc[items['item_id'].isin(recos_data['item_id'])].copy()

        item_feature = items[["item_id", "category_id"]].explode("category_id")
        item_feature.columns = ["id", "value"]
        item_feature["feature"] = "category_id"

        recs_dataset = Dataset.construct(
            interactions_df=recos_data[['user_id', 'item_id', 'weight', 'datetime']],
            item_features_df=item_feature,
            cat_item_features=["category_id"],
        )

        self.rec_model = self.make_base_model(factors=256, regularization=0.2, alpha=100)
        self.rec_model.fit(recs_dataset)
        log.info('Recommendation model has been fitted')
        return recs_dataset
