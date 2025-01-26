import pandas as pd
from pandas import DataFrame
import numpy as np
#from datetime import datetime, timedelta
import datetime
from datetime import timedelta
import os
from copy import deepcopy

from model_management.Ensemble_model import Ensemble_Model
from utils.sun_light import calculate_all_day_sunlight
from utils.holidays import *
from utils.prepare_data import read_historical_power_data, convert_weather_obseravtion_data,\
      add_date_related_information, read_historical_forecast_data, get_period_agg_power
from utils.station_info import *

from Pytorch_models.metrics import Array_Metrics
MAE = Array_Metrics.mae
R2_score = Array_Metrics.r2

data_path = './historical/data/'
train_model_main_path = './trained_model_parameters/'
meta_path = './trained_model_parameters/model_meta_2024-08-11/'
model_path = train_model_main_path + f'latest_model/'

time_description_dict = {
    '午後': 12,
    '下午': 15,
    '傍晚': 18
}


aggregation_func_dict = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


def SunLightRate_to_SunFlux(rate: float, station: str, date: datetime) -> float:
    site = site_location_dict[station]
    relative_sun_flux = calculate_all_day_sunlight(site, date)
    sun_flux = relative_sun_flux * rate * 2.5198 / 100 + 8.6888
    return sun_flux


def convert_weather_df(weather_df: DataFrame, data_path: str =data_path) -> DataFrame:
    weather_df = convert_weather_obseravtion_data(weather_df)
    weather_df = add_date_related_information(weather_df, daytime_fn=data_path+'daytime/daytime.csv')
    power_df = read_historical_power_data(data_path + 'power/power_generation_data.csv')
    power_agg_df = get_period_agg_power(power_df, agg_type='max', date_list=list(weather_df['日期']), return_agg_only=True)
    weather_df = pd.merge(weather_df, power_agg_df, on='日期')
    return weather_df


def convert_forecast_data(input_forecast_df: DataFrame) -> DataFrame:
    forecast_df = deepcopy(input_forecast_df)
    forecast_df['站名'] = [town_and_station[town] for town in forecast_df['鄉鎮']]
    
    column_map = {}
    for col in forecast_df.columns:
        if '_' in col:
            column_map[col] = col.replace('_', '預報_')
    forecast_df.rename(column_map, axis=1, inplace=True)
    return forecast_df


def predict_weather_features(model_path: str,
                             input_forecast_df: DataFrame,
                             wind_speed_naive: bool = False) -> DataFrame:
    forecast_df = convert_forecast_data(input_forecast_df) 
    output_df = deepcopy(forecast_df[['日期', '站名']])
    
    Y_feature_list = [
        '日照率', '最低氣溫', '最高氣溫', '氣溫', '風速',
        '午後平均氣溫', '下午平均氣溫', '傍晚平均氣溫',
        '午後平均風速', '下午平均風速', '傍晚平均風速',
        ]
    
    wind_speed_naive_col_dict = {
        '風速': {'cols': [f'風速預報_{hr}' for hr in range(0, 24, 3)], 'func': 'mean'},
        '午後平均風速': {'cols': [f'風速預報_{hr}' for hr in range(12, 18, 3)], 'func': 'mean'},
        '下午平均風速': {'cols': [f'風速預報_{hr}' for hr in range(15, 21, 3)], 'func': 'mean'},
        '傍晚平均風速': {'cols': [f'風速預報_{hr}' for hr in range(18, 24, 3)], 'func': 'mean'},
    }
    
    for Y_feature in Y_feature_list:
        if wind_speed_naive and Y_feature in wind_speed_naive_col_dict.keys():
            this_agg_func = aggregation_func_dict[wind_speed_naive_col_dict[Y_feature]['func']]
            Y_pred = this_agg_func(forecast_df[wind_speed_naive_col_dict[Y_feature]['cols']], axis=1)
        else:
            MODEL = Ensemble_Model(Y_feature, model_path)
            Y_pred = MODEL.predict(forecast_df)
        output_df.loc[:,Y_feature] = Y_pred

    sun_flux = []
    for i in range(len(output_df)):
        this_sun_flux = SunLightRate_to_SunFlux(output_df['日照率'].iloc[i], output_df['站名'].iloc[i], output_df['日期'].iloc[i])
        sun_flux.append(this_sun_flux)
    output_df['全天空日射量'] = sun_flux

    return output_df


def predict_power_features(
        model_path: str,
        input_weather_df: DataFrame,
        data_path: str,
        ) -> DataFrame:
    weather_df = convert_weather_df(input_weather_df,
                                    data_path=data_path)
    output_df = deepcopy(weather_df[['日期']])
    
    Y_feature_list = ['風力', '太陽能', '尖峰負載']
    
    for Y_feature in Y_feature_list:
        MODEL = Ensemble_Model(Y_feature, model_path)
        Y_pred = MODEL.predict(weather_df)
        output_df[Y_feature] = Y_pred
    return output_df


def evaluation(data_path: str =data_path, moving_mae_days: int =7) -> DataFrame:
    # 讀取正確答案與預測答案並合併成一張表
    power_df = read_historical_power_data(data_path + 'power/power_generation_data.csv')
    power_pred_df = pd.read_csv(data_path + 'prediction/power.csv')
    ref_df = pd.read_csv(data_path + 'prediction/ref.csv')
    power_pred_df['日期'] = pd.to_datetime(power_pred_df['日期'])
    combined_df = pd.merge(power_df, power_pred_df, on='日期', how='inner', suffixes=['', '_預測'])

    # 計算誤差
    output_col_list = ['日期']
    for y_feature in ['風力', '太陽能', '尖峰負載']:
        moving_mae = []
        for i in range(1, len(combined_df)+1):
            start_row = max(0, i-moving_mae_days)
            end_row = i
            mae = MAE(np.array(combined_df[y_feature].iloc[start_row:end_row]), np.array(combined_df[f'{y_feature}_預測'].iloc[start_row:end_row]))
            moving_mae.append(mae)
        combined_df[f'{y_feature}{moving_mae_days}日內平均誤差'] = moving_mae
        output_col_list += [y_feature, f'{y_feature}_預測', f'{y_feature}{moving_mae_days}日內平均誤差']

    # 整理、儲存與回傳評估表
    combined_df = combined_df[output_col_list]
    combined_df['日期'] = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in combined_df['日期']]
    combined_df.loc[len(combined_df)] = ['歷史標準差', '-', '-', ref_df['歷史標準差'].iloc[0], '-', '-', ref_df['歷史標準差'].iloc[1], '-', '-', ref_df['歷史標準差'].iloc[2]]
    combined_df.to_csv(data_path+'prediction/evaluation.csv', index=False, encoding='utf-8-sig')
    return combined_df


def main_predict(
        data_path: str = data_path,
        model_path: str = model_path,
        predict_days: int = 7,
        save_prediction: bool = True,
        update_prediction: bool = False,
        avoid_training_set: bool = True,
        predict_weather_only: bool = False,
        wind_speed_naive: bool = False,
) -> DataFrame:
    '''主要的預測函數，會產生並儲存天氣與電力預測結果
    Arg:
        data_path (str, optional): 預測用資料路徑
        model_path (str, optional): 預測用模型路徑
        predict_days (int, optional): 最多預測幾天的資料，預設為 7 天
        save_prediction (bool, optional): 是否要將預測資料儲存到 data_path 的指定位置，預設為 True
        update_prediction (bool, optional): 若指定位置的預測資料與新預測資料重複，是否覆寫資料，預設為 False
        avoid_training_set (bool, optional): 是否限制模型訓練集包含的天數不予預測，預設為 True,
        predict_weather_only (bool, optional): 是否只預測氣象數值，預設為 False,
        wind_speed_naive (bool, optional): 直接使用風速預報值的簡單平均來預測風速，不經過模型，預設為 False
    '''
    
    # 準備預測結果存放目錄
    historical_pred_path = data_path + 'prediction/'
    os.makedirs(historical_pred_path, exist_ok=True)
    
    # 讀取預報資料
    forecast_df = read_historical_forecast_data(data_path + 'weather/finalized/weather_forecast.csv')
    
    # 決定從哪一天開始預測
    latest_date = max(forecast_df['日期'])
    first_date = latest_date - timedelta(days=predict_days-1)
    if avoid_training_set:
        # 確保預測資料跟使用模型當初的訓練集不重複
        model_training_df = pd.read_csv(model_path + '太陽能/data.csv')
        model_training_df['日期'] = pd.to_datetime(model_training_df['日期'])
        model_last_date = max(model_training_df['日期'])
        first_date = max(model_last_date + timedelta(days=1), first_date)
    forecast_df = forecast_df[forecast_df['日期'] >= first_date]

    # 定義當新舊資料重複時要留哪一個
    keep = 'first' if update_prediction else 'last'

    # 完成預測
    wDF = predict_weather_features(model_path, input_forecast_df=forecast_df,
                                   wind_speed_naive=wind_speed_naive)
    if predict_weather_only:
        return wDF
    pwd_DF = predict_power_features(model_path, wDF, data_path=data_path)
    wDF['日期'] = wDF['日期'].dt.date
    pwd_DF['日期'] = pwd_DF['日期'].dt.date

    # 儲存天氣預測資料
    weather_prediction_filename = historical_pred_path + 'weather.csv'
    if os.path.exists(weather_prediction_filename):
        old_weather_pred_df = pd.read_csv(weather_prediction_filename)
        old_weather_pred_df['日期'] = pd.to_datetime(old_weather_pred_df['日期']).dt.date
        new_weather_pred_df = wDF
        weather_pred_df = pd.concat([new_weather_pred_df, old_weather_pred_df], axis=0)
        weather_pred_df = weather_pred_df.drop_duplicates(subset=['日期', '站名'], keep=keep)
        weather_pred_df = weather_pred_df.sort_values('日期').reset_index(drop=True)
        if save_prediction:
            weather_pred_df.to_csv(weather_prediction_filename, index=False, encoding='utf-8-sig')
    else:
        wDF = wDF.sort_values('日期').reset_index(drop=True)
        if save_prediction:
            wDF.to_csv(weather_prediction_filename, index=False, encoding='utf-8-sig')

    # 儲存電力預測資料
    power_prediction_filename = historical_pred_path + 'power.csv'
    new_power_pred_df = pwd_DF
    if os.path.exists(power_prediction_filename):
        old_power_pred_df = pd.read_csv(power_prediction_filename)
        old_power_pred_df['日期'] = pd.to_datetime(old_power_pred_df['日期']).dt.date
        power_pred_df = pd.concat([new_power_pred_df, old_power_pred_df], axis=0)
        power_pred_df = power_pred_df.drop_duplicates(subset='日期', keep=keep)
        power_pred_df = power_pred_df.sort_values('日期').reset_index(drop=True)
        if save_prediction:
            power_pred_df.to_csv(power_prediction_filename, index=False, encoding='utf-8-sig')
    else:
        pwd_DF = pwd_DF.sort_values('日期').reset_index(drop=True)
        if save_prediction:
            pwd_DF.to_csv(power_prediction_filename, index=False, encoding='utf-8-sig')

    # 儲存歷史電力資料標準差
    ref_filename = historical_pred_path + 'ref.csv'
    if not os.path.exists(ref_filename):
        ref_dict = {'預測項目':[], '歷史平均':[], '歷史標準差':[]}
        for y_feature in ['風力', '太陽能', '尖峰負載']:
            model_data_df = pd.read_csv(model_path + f'{y_feature}/data.csv')
            ref_dict['預測項目'].append(y_feature)
            ref_dict['歷史平均'].append(np.mean(model_data_df[y_feature]))
            ref_dict['歷史標準差'].append(np.std(model_data_df[y_feature]))
        ref_df = DataFrame(ref_dict)
        ref_df.to_csv(ref_filename, index=False, encoding='utf-8-sig')
    
    return new_power_pred_df


if __name__ == '__main__':
    _ = main_predict()


