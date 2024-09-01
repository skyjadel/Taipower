import pandas as pd
import numpy as np
import datetime
import os
from copy import deepcopy

from model_management.Ensemble_model import Ensemble_Model
from utils.sun_light import calculate_all_day_sunlight
from utils.holidays import *
from utils.prepare_data import read_historical_power_data
from utils.sun_light import calculate_daytime

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

def fill_na(df):
    nan_series = df.isna().sum()
    for col in df.columns:
        if nan_series[col] > 0:
            df.fillna({col: np.nanmean(df[col])}, inplace=True)
    return df

def convert_weather_df(input_weather_df):
    input_weather_df['日期'] = pd.to_datetime(input_weather_df['日期'])
    
    station_names = ['臺北', '高雄', '嘉義', '東吉島', '臺中電廠']
    col_map = {col: col.split('(')[0] for col in input_weather_df.columns}

    big_weather_df = deepcopy(input_weather_df)
    big_weather_df.rename(columns=col_map, inplace=True)

    ## 把天氣觀測資料轉成數字格式
    for i in big_weather_df.index:
        for col in big_weather_df.columns:
            if not col in ['站名', '日期']:
                try:
                    big_weather_df.loc[i, col] = np.float32(big_weather_df.loc[i, col])
                except:
                    if big_weather_df.loc[i, col] == 'T':
                        big_weather_df.loc[i, col] = 0
                    else:
                        big_weather_df.loc[i, col] = np.nan

    if ('風速' in big_weather_df.columns and '風向' in big_weather_df.columns)\
        and not ('東西風' in big_weather_df.columns and '南北風' in big_weather_df.columns):
        wind_speed = list(big_weather_df['風速'])
        wind_direction = list(big_weather_df['風向'] / 180 * np.pi)
        NS_wind = np.abs(wind_speed * np.cos(wind_direction))
        EW_wind = np.abs(wind_speed * np.sin(wind_direction))
        big_weather_df['東西風'] = EW_wind
        big_weather_df['南北風'] = NS_wind

        big_weather_df.drop('風向', axis=1, inplace=True)

    if ('最大瞬間風' in big_weather_df.columns and '最大瞬間風風向' in big_weather_df.columns)\
        and not ('東西陣風' in big_weather_df.columns and '南北陣風' in big_weather_df.columns):
        wind_speed = list(big_weather_df['最大瞬間風'])
        wind_direction = list(big_weather_df['最大瞬間風風向'] / 180 * np.pi)
        NS_wind = np.abs(wind_speed * np.cos(wind_direction))
        EW_wind = np.abs(wind_speed * np.sin(wind_direction))
        big_weather_df['東西陣風'] = EW_wind
        big_weather_df['南北陣風'] = NS_wind

        big_weather_df.drop('最大瞬間風風向', axis=1, inplace=True)

    ## 欄名轉換成 {站名}_{觀測值} 的模式
    w_dfs = {}
    for station in station_names:
        this_df = big_weather_df[big_weather_df['站名'] == station]
        w_dfs[station] = this_df.reset_index(drop=True)
    date_column = w_dfs[station_names[0]]['日期']
    for station in station_names:
        w_dfs[station] = w_dfs[station].drop(['日期', '站名'], axis=1)

    weather_df = pd.concat([date_column] + [w_dfs[station].add_suffix(f'_{station}') for station in station_names], axis=1)
    weather_df = fill_na(weather_df)

    # 日期數字化
    date_num = []
    first_date = pd.Timestamp(datetime.date(2023, 8, 1))
    for i in range(len(weather_df)):
        this_date = weather_df['日期'].iloc[i]
        date_num.append((this_date - first_date)/datetime.timedelta(days=1))
    weather_df['日期數字'] = date_num            

    # 加入假日與工作日變量
    weather_df['假日'] = [1 if d in holidays else 0 for d in weather_df['日期']]
    weather_df['週六'] = [1 if d.weekday() == 5 else 0 for d in weather_df['日期']]
    weather_df['週日'] = [1 if d.weekday() == 6 else 0 for d in weather_df['日期']]
    weather_df['補班'] = [1 if d.weekday() in adjusted_work_days else 0 for d in weather_df['日期']]


    # 加入季節變量
    weather_df['1~3月'] = [1 if d.month in [1, 2, 3] and np.sum(weather_df[['假日', '週六', '週日']].iloc[i]) == 0 else 0\
                                for i, d in enumerate(weather_df['日期'])]
    weather_df['11~12月'] = [1 if d.month in [11, 12] and np.sum(weather_df[['假日', '週六', '週日']].iloc[i]) == 0 else 0\
                                for i, d in enumerate(weather_df['日期'])]
    
    # 加入白天長度
    site = {
    'lon': '123.00',
    'lat': '23.5',
    'elevation': 0
    }
    weather_df['白日長度'] = [calculate_daytime(site, date) for date in weather_df['日期']]

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

def predict_weather_features(model_path, input_forecast_df):
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


def predict_power_features(model_path, input_weather_df):
    weather_df = convert_weather_df(input_weather_df)
    output_df = deepcopy(weather_df[['日期']])
    
    Y_feature_list = ['風力', '太陽能', '尖峰負載']
    
    for Y_feature in Y_feature_list:
        MODEL = Ensemble_Model(Y_feature, model_path)
        Y_pred = MODEL.predict(weather_df)
        output_df[Y_feature] = Y_pred
    return output_df

def evaluation(data_path=data_path, moving_mae_days=7):
    power_df = read_historical_power_data(data_path + 'power/power_generation_data.csv')
    power_pred_df = pd.read_csv(data_path + 'prediction/power.csv')
    ref_df = pd.read_csv(data_path + 'prediction/ref.csv')
    power_pred_df['日期'] = pd.to_datetime(power_pred_df['日期'])
    combined_df = pd.merge(power_df, power_pred_df, on='日期', how='inner', suffixes=['', '_預測'])
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
    combined_df = combined_df[output_col_list]
    combined_df['日期'] = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in combined_df['日期']]
    combined_df.loc[len(combined_df)] = ['歷史標準差', '-', '-', ref_df['歷史標準差'].iloc[0], '-', '-', ref_df['歷史標準差'].iloc[1], '-', '-', ref_df['歷史標準差'].iloc[2]]
    combined_df.to_csv(data_path+'prediction/evaluation.csv', index=False, encoding='utf-8-sig')
    return combined_df

def main_predict(data_path=data_path, model_path=model_path,
                 predict_days=7, save_prediction=True, update_prediction=False, avoid_training_set=True):
    
    forecast_df = pd.read_csv(data_path + 'weather/finalized/weather_forecast.csv')
    forecast_df['日期'] = pd.to_datetime(forecast_df['日期'])
    latest_date = max(forecast_df['日期'])
    first_date = latest_date - datetime.timedelta(days=predict_days-1)
    
    # Avoid doing predictions for dates which are included in the training set of the model.
    if avoid_training_set:
        model_training_df = pd.read_csv(model_path + '太陽能/data.csv')
        model_training_df['日期'] = pd.to_datetime(model_training_df['日期'])
        model_last_date = max(model_training_df['日期'])
        first_date = max(model_last_date + datetime.timedelta(days=1), first_date)
        
    forecast_df = forecast_df[forecast_df['日期'] >= first_date]

    wDF = predict_weather_features(model_path, input_forecast_df=forecast_df)
    pwd_DF = predict_power_features(model_path, wDF)
    wDF['日期'] = wDF['日期'].dt.date
    pwd_DF['日期'] = pwd_DF['日期'].dt.date
    
    historical_pred_path = data_path + 'prediction/'
    os.makedirs(historical_pred_path, exist_ok=True)

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

    ref_filename = historical_pred_path + 'ref.csv'
    if not os.path.exists(ref_filename):
        ref_dict = {'預測項目':[], '歷史平均':[], '歷史標準差':[]}
        for y_feature in ['風力', '太陽能', '尖峰負載']:
            training_df = pd.read_csv(model_path + f'{y_feature}/data.csv')
            ref_dict['預測項目'].append(y_feature)
            ref_dict['歷史平均'].append(np.mean(training_df[y_feature]))
            ref_dict['歷史標準差'].append(np.std(training_df[y_feature]))
        ref_df = pd.DataFrame(ref_dict)
        ref_df.to_csv(ref_filename, index=False, encoding='utf-8-sig')
    
    return new_power_pred_df

if __name__ == '__main__':
    _ = main_predict()


