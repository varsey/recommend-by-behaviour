from pathlib import Path

from src.stages import etl
from src.stages.base_filtering import get_test_data, get_core_candidates
from src.stages.prediction import filter_data_to_train, prepare_data_to_predict, predict_purchasers
from src.stages.add_recos import enrich_preds_with_recos
from src.stages.add_similar import add_similar_items

from src.tools.classificator import Classificator
from src.tools.estimator import estimate_recall
from src.tools.rectool import RecTool

from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log


if __name__ == '__main__':
    catalog_path = f'{Path.cwd()}/data/stokman_catalog_preprocessed.pq'
    actions_path = f'{Path.cwd()}/data/train_actions.pq'
    interactions, items, catalog = etl.load_data(catalog_path, actions_path)

    inters = etl.generate_features(interactions, items)
    cl = Classificator()
    cl.train_model(inters, shift_days=9, test_days = 3)

    test_purchasers = get_test_data(inters, start_days=3, end_days=0)
    core_cand = get_core_candidates(inters, days_to_shift=3)
    estimate_recall(test_purchasers, core_cand)

    data_to_predict_on = filter_data_to_train(inters, late_days = 3, earliest_days= 27)
    predictions = prepare_data_to_predict(data_to_predict_on, cl.gbm_model, cl.col)
    combined_preds = predict_purchasers(predictions, test_purchasers, core_cand)
    estimate_recall(test_purchasers, combined_preds)

    rt = RecTool()
    recs_dataset = rt.fit_recommends(catalog, data_to_predict_on)
    enriched_data = enrich_preds_with_recos(rt.rec_model, combined_preds, recs_dataset)
    estimate_recall(test_purchasers, enriched_data)

    enriched_data = add_similar_items(catalog, enriched_data)
    estimate_recall(test_purchasers, enriched_data)

    enriched_data.rename(columns={'item_id': 'products'}).reset_index(drop=True).to_csv("submit.csv", index=False)

    cl.shutdown_h2o()
    log.info('Done')
