import numpy as np
import pandas as pd
import datetime
import os
import joblib
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, NuSVR

from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import f1_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from Pytorch_models.metrics import Array_Metrics
from Pytorch_models import models as pytorch_models
from Pytorch_models import api
MAE = Array_Metrics.mae
R2_score = Array_Metrics.r2

from utils.prepare_data import prepare_data, prepare_forecast_observation_df

#historical_data_path = '../historical/data/'

class Ensemble_Model():
    '''預測某個 feature (i.e.: 太陽能) 的模型集成 API
    Args:
        Y_feature (str): 預測 feature
        model_path (str, optional): 現成模型的位置，若有提供則從裡面讀取模型
        X_feature_dict (dict, optional): 集成學習的每個模型輸入的特徵列表
        hyperparameters_dict (dict, optional): 每個模型的超參數
        weights (dict or str, optional): 集成學習的權重
        data_path (str, optional): 訓練模型用資料的路徑
        start_date (str, optional): 訓練集資料的開始時間，若早於資料庫的最早時間則以資料庫為準
        end_date (str, optional): 訓練集資料的結束時間，若早於資料庫的結束時間則以資料庫為準
        test_size (float, optional): 測試集比例，0~1之間的浮點數，預設為 0.2
        test_last_fold (bool, optional): 只以最新資料做為測試集，預設為 False
        apply_night_peak (bool, optional): 是否採取夜尖峰修正，只有在 Y_feature 為太陽能的時候有影響，預設為 False
        NP_X_feature_dict (dict, optional): 夜尖峰模型使用的特徵，只有在 Y_feature 為太陽能，而且 apply_night_peak 為 True 時會用到
        NP_hyperparameters_dict (dict, optional): 夜尖峰模型使用的超參數，只有在 Y_feature 為太陽能，而且 apply_night_peak 為 True 時會用到
        NP_weights (dict, optional): 夜尖峰模型使用的權重，只有在 Y_feature 為太陽能，而且 apply_night_peak 為 True 時會用到
        remove_night_peak_samples (bool, optional): 訓練時是否排除夜尖峰樣本，只有在 Y_feature 為太陽能的時候有影響，預設為 True

    Main Methods:
        train(): 訓練模型，不需另外輸入參數
        predict(): 利用模型做預測，需要輸入一個 pandas DataFrame 包含模型所需的所有特徵，回傳預測值
        save_model(): 儲存模型參數，需要輸入儲存路徑，若路徑不存在會自動建立
        load_model(): 讀取模型參數，需要輸入模型檔案路徑
    '''
    def __init__(self, Y_feature, 
                 model_path='None', 
                 X_feature_dict=None, hyperparameters_dict=None, weights='uniform',
                 data_path=None, start_date='2023-08-01', end_date='2024-09-30',
                 test_size=0.2, test_last_fold=False,
                 apply_night_peak=False, NP_X_feature_dict=None, NP_hyperparameters_dict=None, NP_weights=None,
                 remove_night_peak_samples=True):

        self.station_list = ['臺北', '高雄', '嘉義', '東吉島', '臺中電廠']
        self.weather_features = ['氣溫', '最高氣溫', '最低氣溫', '風速', '全天空日射量', '總雲量', '東西風', '南北風']
        self.forecast_features = ['晴', '多雲', '陰', '短暫陣雨', '短暫陣雨或雷雨', '午後短暫雷陣雨', '陣雨或雷雨',
                                  '溫度', '降水機率', '相對溼度', '風速', '東西風', '南北風']
        self.single_column_names = ['日期數字', '假日', '週六', '週日', '補班', '1~3月', '11~12月']

        self.Y_feature = Y_feature
        self.apply_night_peak = apply_night_peak
        self.remove_night_peak_samples = remove_night_peak_samples
        self.X_feature_dict = X_feature_dict
        self.hyperparameters_dict = hyperparameters_dict
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = data_path
        self.test_size = test_size
        self.test_last_fold = test_last_fold

        if self.Y_feature in ['風力', '太陽能', '尖峰負載', '日照率', '最高氣溫', '最低氣溫', '氣溫', '風速']:
            self.mode = 'regressor'
            self.varify_metric = R2_score
        elif self.Y_feature == '夜尖峰':
            self.mode = 'classifier'
            self.varify_metric = f1_score

        if self.Y_feature in ['風力', '太陽能', '尖峰負載', '夜尖峰']:
            self.predict_way = 'obs_to_pwd'
        elif self.Y_feature in ['日照率', '最高氣溫', '最低氣溫', '氣溫', '風速']:
            self.predict_way = 'fore_to_obs'

        # 如果有給 model_path，就從裡面讀取模型
        if not model_path == 'None':
            if os.path.exists(model_path + f'{self.Y_feature}/'):
                self.load_model(model_path + f'{self.Y_feature}/')
            else:
                raise ValueError('model_path does not exist.')
            return None

        if self.predict_way == 'obs_to_pwd':
            self.data_df = prepare_data(self.data_path, start_date=self.start_date, end_date=self.end_date)
        elif self.predict_way == 'fore_to_obs':
            self.data_df = prepare_forecast_observation_df(self.data_path, start_date=self.start_date, end_date=self.end_date)
            
        self.model_labels = list(self.X_feature_dict.keys())
        self.models = {model_label: self.assign_model(model_label) for model_label in self.model_labels}
                     
        self.scalers = {}
        self.X_cols = {}
        # for model_label in self.model_labels:
        #     self.models[model_label] = self.assign_model(model_label)
        self.weights = weights
        if weights == 'uniform':
            self.weights = {label: 1/len(self.model_labels) for label in self.model_labels}

        if self.Y_feature in ['太陽能', '夜尖峰']:
            self.data_df['夜尖峰'] = [0 if se > 50 else 1 for se in self.data_df['太陽能']]

        if self.Y_feature == '太陽能' and self.apply_night_peak:
            self.Night_Peak_Model = Ensemble_Model(Y_feature='夜尖峰',
                                                   X_feature_dict=NP_X_feature_dict,
                                                   hyperparameters_dict=NP_hyperparameters_dict,
                                                   weights=NP_weights,
                                                   data_path=data_path,
                                                   start_date=start_date,
                                                   end_date=end_date)      
        
    # Define Fully Connected Network Model
    def FCN_model(self, input_f, output_f, feature_counts=[16, 16, 16, 8], feature_count_label=None, dropout_factor=0, L2_factor=1e-15, mode='regressor'):
        feature_count_dict = {
            'A': [16, 16, 16, 8],
            'B': [24, 16, 16, 8]
        }
        
        if not feature_count_label is None:
            if type(feature_count_label) in [type, tuple]:
                feature_counts = list(feature_count_label)
            else:
                feature_counts = feature_count_dict[feature_count_label]
                
        if mode == 'regressor':
            model = pytorch_models.SimpleNN(input_f, output_f, feature_counts, dropout_factor)
        elif mode == 'classifier':
            model = pytorch_models.SimpleNN_classifer(input_f, output_f, feature_counts, dropout_factor)
        Model_API = api.Model_API(model, L2_factor=L2_factor, classifer=(mode=='classifier'))
        return Model_API

    def fill_nan(self, X):
        nan_idx = np.where(np.isnan(X))
        for i in range(nan_idx[0].shape[0]):
            ri = nan_idx[0][i]
            ci = nan_idx[1][i]
            X[ri][ci] = np.nanmean(X[:,ci])
        return X

    def get_XY(self, data_df, Y_feature, X_features=None):
        station_list = self.station_list    
        X_cols = []

        if X_features is None:
            if self.predict_way == 'obs_to_pwd':
                X_features=self.weather_features + self.single_column_names
            elif self.predict_way == 'fore_to_obs':
                X_features=self.forecast_features

        # 定義 X 使用的欄位名
        for x_f in X_features:
            if self.predict_way == 'obs_to_pwd':
                if x_f in data_df.columns:
                    X_cols.append(x_f)
                    continue
                for station in station_list:
                    this_col = f'{x_f}_{station}'
                    if this_col in data_df.columns:
                        X_cols.append(this_col)

            elif self.predict_way == 'fore_to_obs':
                for hr in range(0, 24, 3):
                    this_col = f'{x_f}預報_{hr}'
                    if this_col in data_df.columns:
                        X_cols.append(this_col)
    
        Xs = np.array(data_df[X_cols])
        Ys = np.array(data_df[Y_feature])

        # 缺失值處理
        Xs = Xs[np.where(~np.isnan(Ys))]
        Ys = Ys[np.where(~np.isnan(Ys))]
        Xs = self.fill_nan(Xs)
            
        return Xs, Ys, X_cols

    def get_train_and_test_index(self, n_samples, test_size=0.2, test_last_fold=False):
        shuffle = not test_last_fold
        train_idx, test_idx, _, _ = train_test_split(np.arange(n_samples), np.arange(n_samples), test_size=test_size, shuffle=shuffle)
        return train_idx, test_idx

    def get_train_and_test_data(self, Xs, Ys, test_size=0.2, test_last_fold=False):
        shuffle = not test_last_fold
        X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys, test_size=test_size, shuffle=shuffle)
        return X_train, X_test, Y_train, Y_test

    def calculate_input_x_feature_num(self, model_label):
        Y_feature = '太陽能' if self.Y_feature == '夜尖峰' else self.Y_feature
        Xs, _, _ = self.get_XY(self.data_df, Y_feature=Y_feature, X_features=self.X_feature_dict[model_label])
        return Xs.shape[1]

    def assign_model(self, model_label):
        if self.mode == 'classifier':
            if model_label == 'RandomForest':
                model = RandomForestClassifier(**self.hyperparameters_dict[model_label])
            elif model_label == 'XGBoost':
                model = XGBClassifier(**self.hyperparameters_dict[model_label])
            elif model_label == 'LightGBM':
                model = LGBMClassifier(force_col_wise=True, verbose=-1,
                                       **self.hyperparameters_dict[model_label])
            elif model_label == 'SVC':
                model = SVC(**self.hyperparameters_dict[model_label])
            elif model_label == 'NuSVC':
                model = NuSVC(**self.hyperparameters_dict[model_label])
            elif model_label == 'LogisticRegression':
                model = LogisticRegression(**self.hyperparameters_dict[model_label])
            elif model_label == 'FCN':
                input_f = self.calculate_input_x_feature_num(model_label)
                output_f = 1
                feature_counts = [16, 16, 16, 8]
                model = self.FCN_model(input_f=input_f, output_f=output_f, feature_counts=feature_counts,
                                       mode='classifier', **self.hyperparameters_dict[model_label])
            else:
                raise ValueError(f'model_label "{model_label}" is not in preset list.')
            return model
        if self.mode == 'regressor':
            if model_label == 'LinearRegression':
                model = LinearRegression()
            elif model_label == 'RandomForest':
                model = RandomForestRegressor(**self.hyperparameters_dict[model_label])
            elif model_label == 'XGBoost':
                model = XGBRegressor(**self.hyperparameters_dict[model_label])
            elif model_label == 'LightGBM':
                model = LGBMRegressor(force_col_wise=True, verbose=-1, **self.hyperparameters_dict[model_label])
            elif model_label == 'SVR':
                model = SVR(**self.hyperparameters_dict[model_label])
            elif model_label == 'NuSVR':
                model = NuSVR(**self.hyperparameters_dict[model_label])
            elif model_label == 'FCN':
                input_f = self.calculate_input_x_feature_num(model_label)
                output_f = 1
                feature_counts = [16, 16, 16, 8]
                model = self.FCN_model(input_f=input_f, output_f=output_f, feature_counts=feature_counts, 
                                       **self.hyperparameters_dict[model_label])
            else:
                raise ValueError(f'model_label "{model_label}" is not in preset list.')
            return model

    def save_model_metadata(self, file_path):
        output_dict = {
            'X_feature_dict':{},
            'hyperparameters_dict':{},
            'weights':{}
        }
        for model_label in self.model_labels:
            if self.weights[model_label] > 0.0005:
                output_dict['X_feature_dict'][model_label] = self.X_feature_dict[model_label]
                output_dict['hyperparameters_dict'][model_label] = self.hyperparameters_dict[model_label]
                output_dict['weights'][model_label] = self.weights[model_label]
    
        with open(file_path, 'w') as f:
            json.dump(output_dict, f)

    def load_model_metadata(self, file_path):
        with open(file_path, 'r') as f:
            meta = json.load(f)
        self.X_feature_dict = meta['X_feature_dict']
        self.hyperparameters_dict = meta['hyperparameters_dict']
        self.weights = meta['weights']

    def train_ML_model(self, model_label, X_train, Y_train):
        
        self.scalers[model_label] = StandardScaler().fit(X_train)
        X_train = self.scalers[model_label].transform(X_train)
        
        _ = self.models[model_label].fit(X_train, Y_train)

    def train_DL_model(self, model_label, X_train, Y_train):

        X_train, X_val, Y_train, Y_val = self.get_train_and_test_data(X_train, Y_train, test_size=0.2, test_last_fold=False)
        
        self.scalers[model_label] = StandardScaler().fit(X_train)
        X_train = self.scalers[model_label].transform(X_train)
        X_val = self.scalers[model_label].transform(X_val)
        
        _ = self.models[model_label].fit(X_train, Y_train, X_val, Y_val)

    def train(self):
        if not hasattr(self, 'train_ind'):
            self.train_ind, self.test_ind, _, _ = train_test_split(np.arange(self.data_df.shape[0]), np.arange(self.data_df.shape[0]), 
                                                                   test_size=self.test_size, shuffle=(not self.test_last_fold))
        self.train_df = self.data_df.iloc[self.train_ind]
        self.train_df = self.train_df[~self.train_df[self.Y_feature].isna()]
        if self.Y_feature == '太陽能' and self.remove_night_peak_samples:
            self.train_df = self.train_df[self.train_df['夜尖峰']==0]
            
        for i, model_label in enumerate(self.model_labels):
            print(f'({i+1}/{len(self.model_labels)}) {model_label}')
            X_train, Y_train, X_cols = self.get_XY(self.train_df, Y_feature=self.Y_feature, X_features=self.X_feature_dict[model_label])
            self.X_cols[model_label] = X_cols

            if model_label == 'FCN':
                self.train_DL_model(model_label, X_train, Y_train)
            else:
                self.train_ML_model(model_label, X_train, Y_train)

        if self.Y_feature == '太陽能' and self.apply_night_peak:
            self.Night_Peak_Model.train_ind = self.train_ind
            self.Night_Peak_Model.test_ind = self.test_ind
            self.Night_Peak_Model.train()

    def get_one_prediction(self, df, model_label):
        X = np.array(df[self.X_cols[model_label]])
        X = self.fill_nan(X)
        X = self.scalers[model_label].transform(X)
        return self.models[model_label].predict(X)

    def predict(self, df, return_all_predictions=False, use_model='Ensemble'):
        Y_preds, weights = [], []
        if self.Y_feature == '太陽能' and self.apply_night_peak:
            night_peak = self.Night_Peak_Model.predict(df)
            day_peak = 1 - np.array(night_peak)
        if use_model == 'Ensemble':
            for model_label in self.model_labels:
                Y_P = self.get_one_prediction(df, model_label)
                if self.Y_feature == '太陽能' and self.apply_night_peak:
                    Y_P *= day_peak
                Y_preds.append(Y_P)
                weights.append(self.weights[model_label])
    
            final_prediction = Y_preds[0] * 0.0
            for i, Y_P in enumerate(Y_preds):
                final_prediction += Y_P * weights[i]
    
            if self.mode == 'classifier':
                final_prediction[np.where(final_prediction>=0.5)] = 1
                final_prediction[np.where(final_prediction<0.5)] = 0
                
            if return_all_predictions:
                return final_prediction, Y_preds
            return final_prediction
            
        elif use_model in self.model_labels:
            return self.get_one_prediction(df, use_model)
        else:
            raise ValueError(f'The string "{use_model}" is not included in the model labels. Must choose from {list(self.model_labels)} or "Ensemble".')

    def varify(self, return_var_df=False, var_df=None):
        if var_df is None:
            var_df = self.data_df.iloc[self.test_ind].reset_index(drop=True)
        var_df = var_df[~var_df[self.Y_feature].isna()]

        final_prediction, Y_preds = self.predict(var_df, return_all_predictions=True)
        Y_truth = np.array(var_df[self.Y_feature])
        scores = [self.varify_metric(Y_truth, final_prediction)]
        for i in range(len(Y_preds)):
            scores.append(self.varify_metric(Y_truth, Y_preds[i]))
        if self.mode == 'regressor':
            result_df = pd.DataFrame({'Model': ['Ensemble'] + self.model_labels, 'R2': scores})
        elif self.mode == 'classifier':
            result_df = pd.DataFrame({'Model': ['Ensemble'] + self.model_labels, 'F1': scores})
        if return_var_df:
            return result_df, var_df
        return result_df

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        self.save_model_metadata(model_path + 'meta.json')
        self.train_df.to_csv(model_path + 'training_set.csv', index=False, encoding='utf-8-sig')
        self.data_df.to_csv(model_path + 'data.csv', index=False, encoding='utf-8-sig')
        
        for model_label in self.model_labels:
            this_path = model_path + f'{model_label}/'
            os.makedirs(this_path, exist_ok=True)
            
            if model_label == 'FCN':
                self.models[model_label].save_weight(this_path + 'model/')
            else:
                joblib.dump(self.models[model_label], this_path + 'model.pkl')

            joblib.dump(self.scalers[model_label], this_path + 'XScaler.pkl')
            
            with open(this_path + 'X_columns.txt', 'w', encoding='utf-8-sig') as f:
                for col in self.X_cols[model_label]:
                    f.write(col + ', ')

        if self.apply_night_peak and self.Y_feature == '太陽能':
            NP_path = model_path + 'NP_model/'
            os.makedirs(NP_path, exist_ok=True)
            self.Night_Peak_Model.save_model(NP_path)

    def load_model(self, model_path):
        
        def get_x_cols(filename):
            with open(filename, 'r', encoding='utf-8-sig'):
                X_cols_str = f.read()
            return X_cols_str.split(', ')[0:-1]

        self.load_model_metadata(model_path + 'meta.json')
        self.data_df = pd.read_csv(model_path + 'data.csv')
        self.train_df = pd.read_csv(model_path + 'training_set.csv')

        self.model_labels = list(self.X_feature_dict.keys())
        
        self.X_cols = {model_label: get_x_cols(f'{model_path}{model_label}/X_columns.txt') for model_label in self.model_labels}
        self.scalers = {model_label: joblib.load(f'{model_path}{model_label}/XScaler.pkl') for model_label in self.model_labels}
        
        self.models = {}
        for model_label in self.model_labels:
            this_path = model_path + f'{model_label}/'
            if model_label == 'FCN':
                self.models[model_label] = self.assign_model(model_label)
                self.models[model_label].load_weight(this_path + 'model/')
            else:
                self.models[model_label] = joblib.load(this_path + 'model.pkl')

        NP_path = model_path + 'NP_model/'
        if self.Y_feature == '太陽能' and os.path.exist(NP_path):
            self.apply_night_peak = True
            self.Night_Peak_Model = Ensemble_Model(Y_feature='夜尖峰', model_path=NP_path)
