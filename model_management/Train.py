import json
import os
import datetime

from model_management.Ensemble_model import Ensemble_Model
from Pytorch_models.metrics import Array_Metrics
from utils.station_info import effective_station_list

MAE = Array_Metrics.mae
R2_score = Array_Metrics.r2

params_dict = {
    'meta_path': '../trained_model_parameters/model_meta_2024-08-28/', # 讀取 meta parameters 的路徑
    'data_path': '../historical/data/', # 訓練用資料的路徑
    'test_size': 0.001, # 測試集的比例
    'test_last_fold': False, # 是否選取時間最晚近的資料做為測試集
    'apply_night_peak': False, # 太陽能部分是否加入夜尖峰調整
    'start_date': '2023-08-01', # 訓練資料開始日
    'end_date': '2200-12-31' # 訓練資料結束日
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


# 為單一一種被預測值訓練模型
def train_one_model(Y_feature, model_path, meta_path, data_path,
                    start_date='2023-08-01', end_date='2200-12-31', effective_station_list=effective_station_list,
                    test_size=0.2, test_last_fold=True, apply_night_peak=False, remove_night_peak_samples=True, fit_wind_square=False,
                    latest_model_path=latest_model_path):
    with open(f'{meta_path}{Y_feature}/meta.json', 'r') as f:
        meta = json.load(f)
    X_feature_dict = meta['X_feature_dict']
    hyperparameters_dict = meta['hyperparameters_dict']
    weights = meta['weights']
    if 'fit_wind_square' in meta.keys():
        fit_wind_square = meta['fit_wind_square']

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
                               effective_station_list=effective_station_list,
                               start_date=start_date,
                               end_date=end_date,
                               test_size=test_size,
                               test_last_fold=test_last_fold,
                               fit_wind_square=fit_wind_square,
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
                               effective_station_list=effective_station_list,
                               start_date=start_date, 
                               end_date=end_date, 
                               test_size=test_size,
                               test_last_fold=test_last_fold,
                               fit_wind_square=fit_wind_square,
                               remove_night_peak_samples=remove_night_peak_samples)
    MODEL.train()

    MODEL.save_model(f'{model_path}{Y_feature}/')
    #save_model_metadata(f'{model_path}{Y_feature}/meta.json', X_feature_dict, hyperparameters_dict, weights)

    if not latest_model_path is None:
        MODEL.save_model(f'{latest_model_path}{Y_feature}/')
        #save_model_metadata(f'{latest_model_path}{Y_feature}/meta.json', X_feature_dict, hyperparameters_dict, weights)


# 訓練一組能夠預測多個特徵的模型
def train_all_models(model_path, meta_path, data_path, effective_station_list=effective_station_list,
                     test_size=0.001, test_last_fold=False,
                     apply_night_peak=False, remove_night_peak_samples=True, latest_model_path=latest_model_path,
                     start_date='2023-08-01', end_date='2200-12-31'):
    YF_list = os.listdir(meta_path)
    for Y_feature in YF_list:
        print(f'{Y_feature} 訓練中.')
        train_one_model(Y_feature, model_path, meta_path, data_path, 
                        effective_station_list=effective_station_list,
                        test_size=test_size, test_last_fold=test_last_fold,
                        apply_night_peak=apply_night_peak, remove_night_peak_samples=remove_night_peak_samples, latest_model_path=latest_model_path,
                        start_date=start_date, end_date=end_date)


def main_train(params=params_dict, preserved_days=0,
               train_model_main_path=train_model_main_path,
               suffix='',
               effective_station_list=effective_station_list,
               apply_night_peak=False, remove_night_peak_samples=True,
               latest_model_path=None):
    '''整合輸入參數，並訓練一組能夠預測多個特徵的模型
    Args:
        params(dict, optional): 部份訓練參數，格式參看模組開頭的預設值
        preserved_days(int, optional): 保留幾天的最新資料不參與訓練，通常是為了事後驗證模型的表現
        train_model_main_path(str, optional): 模型儲存位置
        effective_station_list(List of str, optional): 模型考慮的氣象站列表
        apply_night_peak(bool, optional): 太陽能部分是否加入夜尖峰調整
        remove_night_peak_samples(bool, optional): 太陽能訓練集是否排除夜尖峰樣本
        latest_model_path(str or None, optional): 若不是 None 則自動儲存一份模型參數到指定位置
    '''
    today = datetime.datetime.now().date()
    end_date = today - datetime.timedelta(days=preserved_days+1)
    end_date = datetime.datetime.strftime(end_date, '%Y-%m-%d')
    params['effective_station_list'] = effective_station_list
    params['end_date'] = end_date
    params['apply_night_peak'] = apply_night_peak
    params['remove_night_peak_samples'] = remove_night_peak_samples
    params['latest_model_path'] = latest_model_path

    model_path = train_model_main_path + f'model_{end_date}'
    if apply_night_peak:
        model_path += '_NP'
    if not remove_night_peak_samples:
        model_path += '_PNPS'
    model_path += f'{suffix}/'
    os.makedirs(model_path, exist_ok=True)
    print(model_path)

    train_all_models(model_path=model_path, **params)


if __name__ == '__main__':
    main_train()
