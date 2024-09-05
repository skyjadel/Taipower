import json
import os
import datetime

from model_management.Ensemble_model import Ensemble_Model
from Pytorch_models.metrics import Array_Metrics

MAE = Array_Metrics.mae
R2_score = Array_Metrics.r2

params_dict = {
    'meta_path': '../trained_model_parameters/model_meta_2024-08-28/', 
    'data_path': '../historical/data/', 
    'test_size': 0.001,
    'test_last_fold': False,
    'apply_night_peak': False,
    'start_date': '2023-08-01',
    'end_date': '2200-12-31'
}
train_model_main_path = '../trained_model_parameters/models_tobe_evaluated/'
latest_model_path='../trained_model_parameters/latest_model/'


def save_model_metadata(file_path, model_xcols, model_hyperparameters_dict, optimal_weights):
    model_labels = list(model_hyperparameters_dict)
    output_dict = {
        'X_feature_dict':{},
        'hyperparameters_dict':{},
        'weights':{}
    }
    for model_label in model_labels:
        output_dict['X_feature_dict'][model_label] = model_xcols[model_label]
        output_dict['hyperparameters_dict'][model_label] = model_hyperparameters_dict[model_label]
        output_dict['weights'][model_label] = optimal_weights[model_label]

    with open(file_path, 'w') as f:
        json.dump(output_dict, f)


def train_one_model(Y_feature, model_path, meta_path, data_path, start_date='2023-08-01', end_date='2200-12-31',
                    test_size=0.2, test_last_fold=True, apply_night_peak=False, remove_night_peak_samples=True,
                    latest_model_path=latest_model_path):
    with open(f'{meta_path}{Y_feature}/meta.json', 'r') as f:
        meta = json.load(f)
    X_feature_dict = meta['X_feature_dict']
    hyperparameters_dict = meta['hyperparameters_dict']
    weights = meta['weights']

    normalization_factor = 0
    for weight in weights.values():
        normalization_factor += weight
    for model_label in weights.keys():
        weights[model_label] /= normalization_factor

    
    if Y_feature == '太陽能' and apply_night_peak:
        with open(f'{meta_path}夜尖峰/meta.json', 'r') as f:
            NP_meta = json.load(f)
        NP_X_feature_dict = NP_meta['X_feature_dict']
        NP_hyperparameters_dict = NP_meta['hyperparameters_dict']
        NP_weights = NP_meta['weights']

        MODEL = Ensemble_Model(Y_feature=Y_feature, 
                               X_feature_dict=X_feature_dict, 
                               hyperparameters_dict=hyperparameters_dict, 
                               data_path=data_path, 
                               weights=weights,
                               end_date=end_date,
                               test_size=test_size,
                               test_last_fold=test_last_fold,
                               NP_X_feature_dict=NP_X_feature_dict,
                               NP_hyperparameters_dict=NP_hyperparameters_dict,
                               NP_weights=NP_weights,
                               apply_night_peak=True,
                               remove_night_peak_samples=remove_night_peak_samples)
    else:
        MODEL = Ensemble_Model(Y_feature=Y_feature, 
                               X_feature_dict=X_feature_dict, 
                               hyperparameters_dict=hyperparameters_dict, 
                               data_path=data_path,
                               weights=weights, 
                               start_date=start_date, 
                               end_date=end_date, 
                               test_size=test_size,
                               test_last_fold=test_last_fold,
                               remove_night_peak_samples=remove_night_peak_samples)
    MODEL.train()

    MODEL.save_model(f'{model_path}{Y_feature}/')
    save_model_metadata(f'{model_path}{Y_feature}/meta.json', X_feature_dict, hyperparameters_dict, weights)

    if not latest_model_path is None:
        MODEL.save_model(f'{latest_model_path}{Y_feature}/')
        save_model_metadata(f'{latest_model_path}{Y_feature}/meta.json', X_feature_dict, hyperparameters_dict, weights)

def train_all_models(model_path, meta_path, data_path, test_size=0.001, test_last_fold=False,
                     apply_night_peak=False, remove_night_peak_samples=True, latest_model_path=latest_model_path,
                     start_date='2023-08-01', end_date='2200-12-31'):
    YF_list = os.listdir(meta_path)
    for Y_feature in YF_list:
        print(f'{Y_feature} 訓練中.')
        train_one_model(Y_feature, model_path, meta_path, data_path, test_size=test_size, test_last_fold=test_last_fold,
                        apply_night_peak=apply_night_peak, remove_night_peak_samples=remove_night_peak_samples, latest_model_path=latest_model_path,
                        start_date=start_date, end_date=end_date)

def main_train(params=params_dict, preserved_days=0, train_model_main_path=train_model_main_path,
               apply_night_peak=False, remove_night_peak_samples=True,
               latest_model_path=None):
    today = datetime.datetime.now().date()
    end_date = today - datetime.timedelta(days=preserved_days+1)
    end_date = datetime.datetime.strftime(end_date, '%Y-%m-%d')
    params['end_date'] = end_date
    params['apply_night_peak'] = apply_night_peak
    params['remove_night_peak_samples'] = remove_night_peak_samples
    params['latest_model_path'] = latest_model_path

    model_path = train_model_main_path + f'model_{end_date}'
    if apply_night_peak:
        model_path += '_NP'
    if not remove_night_peak_samples:
        model_path += '_PNPS'
    model_path += '/'
    os.makedirs(model_path, exist_ok=True)
    print(model_path)

    train_all_models(model_path=model_path, **params)


if __name__ == '__main__':
    main_train()
