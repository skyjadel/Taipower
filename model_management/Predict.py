import pandas as pd
import numpy as np
import datetime
import os
from copy import deepcopy

from pandas import DataFrame

from model_management.Ensemble_model import Ensemble_Model
from utils.sun_light import calculate_all_day_sunlight
from utils.holidays import *
from utils.prepare_data import read_historical_power_data, convert_weather_obseravtion_data, add_date_related_information

from Pytorch_models.metrics import Array_Metrics
MAE = Array_Metrics.mae
R2_score = Array_Metrics.r2

data_path = './historical/data/'
train_model_main_path = './trained_model_parameters/'
meta_path = './trained_model_parameters/model_meta_2024-08-11/'
model_path = train_model_main_path + f'latest_model/'

def SunLightRate_to_SunFlux(rate, station, date):
    site_location_dict = {
        '臺北': {'lat':'25.037658' ,'lon':'121.514853', 'elevation':6.26},
        '高雄': {'lat':'22.73043151' ,'lon':'120.3125156', 'elevation':11.79},
        '嘉義': {'lat':'23.495925' ,'lon':'120.4329056', 'elevation':26.9},
        '東吉島': {'lat':'23.25695' ,'lon':'119.6674667', 'elevation':44.5},
        '臺西': {'lat':'23.701544' ,'lon':'120.197547', 'elevation':12},
        '臺中電廠': {'lat':'24.214642' ,'lon':'120.490744', 'elevation':25},
    }
    
    site = site_location_dict[station]
    relative_sun_flux = calculate_all_day_sunlight(site, date)
    sun_flux = relative_sun_flux * rate * 2.5198 / 100 + 8.6888
    return sun_flux

def convert_weather_df(weather_df):
    weather_df = convert_weather_obseravtion_data(weather_df)
    weather_df = add_date_related_information(weather_df)
    return weather_df


def convert_forecast_data(input_forecast_df):
    town_and_station = {
        '臺北市中正區': '臺北',
        '高雄市楠梓區': '高雄',
        '嘉義市西區': '嘉義',
        '澎湖縣望安鄉': '東吉島',
        '臺中市龍井區': '臺中電廠',
        '雲林縣臺西鄉': '臺西'
    }

    forecast_df = deepcopy(input_forecast_df)
    forecast_df['站名'] = [town_and_station[town] for town in forecast_df['鄉鎮']]

    for hr in range(0, 24, 3):
        wind_speed = list(forecast_df[f'風速_{hr}'])
        wind_direction = list(forecast_df[f'風向_{hr}'] / 180 * np.pi)
        NS_wind = np.abs(wind_speed * np.cos(wind_direction))
        EW_wind = np.abs(wind_speed * np.sin(wind_direction))
        forecast_df[f'東西風_{hr}'] = EW_wind
        forecast_df[f'南北風_{hr}'] = NS_wind
        forecast_df.drop([f'風向_{hr}'], axis=1, inplace=True)
    
    column_map = {}
    for col in forecast_df.columns:
        if '_' in col:
            column_map[col] = col.replace('_', '預報_')
    forecast_df.rename(column_map, axis=1, inplace=True)
    return forecast_df

def predict_weather_features(model_path: str, input_forecast_df: DataFrame):
    forecast_df = convert_forecast_data(input_forecast_df) 
    output_df = deepcopy(forecast_df[['日期', '站名']])
    
    Y_feature_list = ['日照率', '最低氣溫', '最高氣溫', '氣溫', '風速']
    
    for Y_feature in Y_feature_list:
        MODEL = Ensemble_Model(Y_feature, model_path)
        Y_pred = MODEL.predict(forecast_df)
        output_df.loc[:,Y_feature] = Y_pred

    sun_flux = []
    for i in range(len(output_df)):
        this_sun_flux = SunLightRate_to_SunFlux(output_df['日照率'].iloc[i], output_df['站名'].iloc[i], output_df['日期'].iloc[i])
        sun_flux.append(this_sun_flux)
    output_df['全天空日射量'] = sun_flux
    return output_df

def predict_power_features(model_path: str, input_weather_df: DataFrame):
    weather_df = convert_weather_df(input_weather_df)
    output_df = deepcopy(weather_df[['日期']])
    
    Y_feature_list = ['風力', '太陽能', '尖峰負載']
    
    for Y_feature in Y_feature_list:
        MODEL = Ensemble_Model(Y_feature, model_path)
        Y_pred = MODEL.predict(weather_df)
        output_df[Y_feature] = Y_pred
    return output_df

def evaluation(data_path=data_path, moving_mae_days=7):
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

def main_predict(data_path: str = data_path,
                 model_path: str = model_path,
                 predict_days: int = 7,
                 save_prediction: bool = True,
                 update_prediction: bool = False,
                 avoid_training_set: bool = True):
    '''主要的預測函數，會產生並儲存天氣與電力預測結果
    Arg:
        data_path (str, optional): 預測用資料路徑
        model_path (str, optional): 預測用模型路徑
        predict_days (int, optional): 最多預測幾天的資料，預設為 7 天
        save_prediction (bool, optional): 是否要將預測資料儲存到 data_path 的指定位置，預設為 True
        update_prediction (bool, optional): 若指定位置的預測資料與新預測資料重複，是否覆寫資料，預設為 False
        avoid_training_set (bool, optional): 是否限制模型訓練集包含的天數不予預測，預設為 True
    '''
    
    # 準備預測結果存放目錄
    historical_pred_path = data_path + 'prediction/'
    os.makedirs(historical_pred_path, exist_ok=True)
    
    # 讀取預報資料
    forecast_df = pd.read_csv(data_path + 'weather/finalized/weather_forecast.csv')
    forecast_df['日期'] = pd.to_datetime(forecast_df['日期'])
    
    # 決定從哪一天開始預測
    latest_date = max(forecast_df['日期'])
    first_date = latest_date - datetime.timedelta(days=predict_days-1)
    if avoid_training_set:
        # 確保預測資料跟使用模型當初的訓練集不重複
        model_training_df = pd.read_csv(model_path + '太陽能/data.csv')
        model_training_df['日期'] = pd.to_datetime(model_training_df['日期'])
        model_last_date = max(model_training_df['日期'])
        first_date = max(model_last_date + datetime.timedelta(days=1), first_date)
        
    forecast_df = forecast_df[forecast_df['日期'] >= first_date]

    # 完成預測
    wDF = predict_weather_features(model_path, input_forecast_df=forecast_df)
    pwd_DF = predict_power_features(model_path, wDF)
    wDF['日期'] = wDF['日期'].dt.date
    pwd_DF['日期'] = pwd_DF['日期'].dt.date

    # 儲存天氣預測資料
    weather_prediction_filename = historical_pred_path + 'weather.csv'
    if os.path.exists(weather_prediction_filename):
        old_weather_pred_df = pd.read_csv(weather_prediction_filename)
        old_weather_pred_df['日期'] = pd.to_datetime(old_weather_pred_df['日期']).dt.date
        new_weather_pred_df = wDF
        weather_pred_df = pd.concat([new_weather_pred_df, old_weather_pred_df], axis=0)
        if update_prediction:
            weather_pred_df = weather_pred_df.drop_duplicates(subset=['日期', '站名'], keep='first')
        else:
            weather_pred_df = weather_pred_df.drop_duplicates(subset=['日期', '站名'], keep='last')
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
        if update_prediction:
            power_pred_df = power_pred_df.drop_duplicates(subset='日期', keep='first')
        else:
            power_pred_df = power_pred_df.drop_duplicates(subset='日期', keep='last')
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
        ref_df = pd.DataFrame(ref_dict)
        ref_df.to_csv(ref_filename, index=False, encoding='utf-8-sig')
    
    return new_power_pred_df

if __name__ == '__main__':
    _ = main_predict()


