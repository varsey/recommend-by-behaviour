import h2o
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator

from src.decorators import duration
from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log


TEST_DF_SIZE = 40_000
SKIP_COLUMNS = ['user_id', 'shop_id', 'day', 'hour', 'pageId', 'datetime', 'item_id', 'is_useful', 'title', 'old_price', 'price']


@duration
class Classificator:
    def __init__(self):
        h2o.init(verbose=False)
        log.info(f'{h2o.cluster().show_status()}')
        h2o.no_progress()
        self.gbm_model = None
        self.col = None

    @staticmethod
    def shutdown_h2o():
        h2o.cluster().shutdown()

    def see_performance(self, test_h2o: h2o.H2OFrame):
        if self.gbm_model:
            self.gbm_model.model_performance(test_h2o)
            self.gbm_model.varimp(use_pandas=True)

    def train_model(self, inters: pd.DataFrame, shift_days: int = 0, test_days = 3):
        max_date = inters['datetime'].max()
        train_raw = inters[inters['datetime'] <= max_date - pd.Timedelta(days=shift_days)]
        test_raw = inters[inters['datetime'] > max_date - pd.Timedelta(days=test_days)].copy()

        train = train_raw.drop(columns=['datetime', 'item_id', 'pageId'])
        test = test_raw.drop(columns=['datetime', 'item_id', 'pageId'])

        train = train[train.weight.isin([0, 4, 3, 6, 7, 9])].drop_duplicates()
        log.info(f'Selected train dataframe size: {train.shape}')

        train_portion = pd.concat(
            [
                train[train.weight.isin([0, 1, 3, 6, 7, 9])],
                train_raw.sample(train.shape[0])],
            axis='rows'
        )

        train_h2o = h2o.H2OFrame(train_portion)
        test_h2o = h2o.H2OFrame(test.sample(TEST_DF_SIZE))

        y = "weight"
        x = set(train_h2o.names) - set([y] + SKIP_COLUMNS)
        self.col = x

        train_h2o[y] = train_h2o[y].asfactor()
        test_h2o[y] = test_h2o[y].asfactor()

        train_h2o['category_id'] = train_h2o['category_id'].asfactor()
        test_h2o['category_id'] = test_h2o['category_id'].asfactor()

        self.gbm_model = H2OGradientBoostingEstimator(seed=1234)
        self.gbm_model.train(x=list(x), y=y, training_frame = train_h2o, max_runtime_secs=120)
