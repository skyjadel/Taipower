# 從整合過的歷史電力與氣象資料，提取模型訓練所需資料，並整理成 DataFrame 格式的模組
# 重要函式：
# prepare_data/prepare_observation_power_df: 產生氣象觀測與電力資料的整合資料表，一天一筆資料，訓練模型所需
# prepare_forecast_observation_df: 產生氣象觀測與氣象預報資料的整合資料表，一天一氣象站一筆資料，訓練模型需要

import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime

from utils.station_info import town_and_station
from utils.convert_wind_info import polar_to_cartesian_coord

import utils.power_generation_types as power_types #各發電機組屬於哪種發電方式的定義檔
from utils.holidays import *
from utils.station_info import effective_station_list as station_names #這邊的氣象站列表只會影響 氣象觀測-電力資訊 的那張表

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 日期數字化的第 0 天
FIRST_DATE_STR = '2023-01-01'

#定義歷史資料的時間範圍
start_date = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')
end_date = datetime.datetime.strptime('2200-06-30', '%Y-%m-%d')


def convert_str_to_datetime(string):
    if not type(string) == str:
        return string
    
    for seperate_str in ['-', '/', '.']:
        if seperate_str in string:
            try:
                return datetime.datetime.strptime(string, f'%Y{seperate_str}%m{seperate_str}%d')
            except:
                return string
    return string


def designate_date_range(df, date_column, start_date, end_date):
    df = df[df[date_column] >= start_date]
    out_df = df[df[date_column] <= end_date].reset_index(drop=True)
    return out_df


def delete_and_fill_na(df):
    nan_series = df.isna().sum()
    delete_columns = []
    for col in nan_series.index:
        if nan_series[col] > 0.4*len(df):
            delete_columns.append(col)
    df.drop(delete_columns, axis=1, inplace=True)

    for col in df.columns:
        if nan_series[col] > 0:
            df.fillna({col: np.nanmean(df[col])}, inplace=True)
    return df

# 在資料表中加入欄位，顯示電力資料到當天為止最近 N 天中的最大值/最小值/平均值/中位數
def get_period_agg_power(
        power_df, power_features=['風力', '太陽能', '尖峰負載'],
        date_list=None, duration=90, agg_type='mean',
        new_date=False, return_agg_only=False,
        ):
    
    agg_func_dict = {
        'max': {'func':np.nanmax, 'suffix':f'_{duration}天內最大值'},
        'min': {'func':np.nanmin, 'suffix':f'_{duration}天內最小值'},
        'mean': {'func':np.nanmean, 'suffix':f'_{duration}天內平均值'},
        'median': {'func':np.nanmedian, 'suffix':f'_{duration}天內中位數'},
    }

    agg_func = agg_func_dict[agg_type]['func']
    agg_suffix = agg_func_dict[agg_type]['suffix']

    if type(power_features) == str:
        power_features = [power_features]

    if date_list is None:
        date_list = list(power_df['日期'])
        if new_date:
            date_list.append(max(date_list) + datetime.timedelta(days=1))

    result_df = DataFrame({'日期': date_list})
    for feature in power_features:
        new_col_name = f'{feature}{agg_suffix}'
        if new_col_name in power_df.columns:
            power_df.drop(new_col_name, axis=1, inplace=True)
        
        temp_date_df = DataFrame({'日期': date_list})
        temp_date_df['日期'] = pd.to_datetime(temp_date_df['日期'])
        operate_df = pd.merge(temp_date_df, power_df, on='日期', how='outer')[['日期', feature]]
        
        result_value_list = []
        for date in date_list:
            start_date = date - datetime.timedelta(days=duration+1)
            end_date = date - datetime.timedelta(days=2)
            this_df = operate_df[(operate_df['日期']>=start_date) & (operate_df['日期']<=end_date)]
            if len(this_df) == 0:
                result_value_list.append(operate_df[operate_df['日期']==date].iloc[0][feature])
                continue
            result_value_list.append(agg_func(this_df[feature]))
        result_df[new_col_name] = result_value_list
    
    if return_agg_only:
        return result_df
    return pd.merge(power_df, result_df, on='日期', how='outer')

# 將颱風假整合成假日
def holiday_identify(date):
    if date in holidays:
        return 1
    if date in typhoon_leave.keys():
        return typhoon_leave[date]
    return 0

# 將儲存的天氣資料表，或模型產生的天氣資料表，轉換成適合輸入模型的格式
def convert_weather_obseravtion_data(input_weather_df, start_date=start_date, end_date=end_date, transform_columns=True):
    input_weather_df['日期'] = pd.to_datetime(input_weather_df['日期'])
    input_weather_df = designate_date_range(input_weather_df, '日期', start_date, end_date)

    col_map = {col: col.split('(')[0] for col in input_weather_df.columns}

    big_weather_df = input_weather_df.copy()
    big_weather_df.rename(columns=col_map, inplace=True)

    ## 把天氣觀測資料轉成數字格式
    for i in big_weather_df.index:
        for col in big_weather_df.columns:
            element = big_weather_df.loc[i, col]
            if element == 'T':
                big_weather_df.loc[i, col] = 0
                continue
            if not col in ['站名', '日期'] and type(element) == str:
                big_weather_df.loc[i, col] = np.nan
    
    # 風向與風速資料轉換        
    if ('風速' in big_weather_df.columns and '風向' in big_weather_df.columns)\
        and not ('東西風' in big_weather_df.columns and '南北風' in big_weather_df.columns):
        big_weather_df = polar_to_cartesian_coord(big_weather_df, ['風速', '風向'], ['東西風', '南北風'])
    
    if ('最大瞬間風' in big_weather_df.columns and '最大瞬間風風向' in big_weather_df.columns)\
        and not ('東西陣風' in big_weather_df.columns and '南北陣風' in big_weather_df.columns):
        big_weather_df = polar_to_cartesian_coord(big_weather_df, ['最大瞬間風', '最大瞬間風風向'], ['東西陣風', '南北陣風'])

    if transform_columns:
        ## 欄名轉換成 {站名}_{觀測值} 的模式
        w_dfs = {station: big_weather_df[big_weather_df['站名'] == station].reset_index(drop=True).drop(['日期', '站名'], axis=1) for station in station_names}
        date_column = big_weather_df[big_weather_df['站名'] == station_names[0]]['日期']
        weather_df = pd.concat([date_column] + [w_dfs[station].add_suffix(f'_{station}') for station in station_names], axis=1)
        weather_df = delete_and_fill_na(weather_df)
        return weather_df
    
    return big_weather_df


def read_historical_forecast_data(data_fn, start_date=start_date, end_date=end_date, transform_columns=False):
    forecast_df = pd.read_csv(data_fn)
    forecast_df['日期'] = pd.to_datetime(forecast_df['日期'])
    forecast_df = designate_date_range(forecast_df, '日期', start_date, end_date)

    # 加入不同氣象站的 one-hot encoding
    labelencoder = LabelEncoder()
    forecast_df['town'] = labelencoder.fit_transform(forecast_df['鄉鎮'])
    onehotencoder = OneHotEncoder()
    data_str_ohe=onehotencoder.fit_transform(forecast_df['town'].to_numpy().reshape(-1,1)).toarray()
    town_onehot_df = DataFrame(data_str_ohe)
    column_mapping = {col: 'Town_' + labelencoder.inverse_transform([col])[0] for col in town_onehot_df.columns}
    town_onehot_df.rename(column_mapping, axis=1, inplace=True)
    forecast_df = pd.concat([forecast_df, town_onehot_df], axis=1).drop('town', axis=1)
    
    for hr in range(0, 24, 3):
        forecast_df[f'風速超過上限_{hr}'] = [1 if s >= 11 else 0 for s in forecast_df[f'風速_{hr}']]
        forecast_df = polar_to_cartesian_coord(forecast_df, [f'風速_{hr}', f'風向_{hr}'], [f'東西風_{hr}', f'南北風_{hr}'])

    if transform_columns:
        town_names = list(town_and_station.keys())
        ## 欄名轉換成 {預報值}_{鐘點}_{鄉鎮名} 的模式
        forecast_dfs = {town: forecast_df[forecast_df['鄉鎮'] == town].reset_index(drop=True).drop(['日期', '鄉鎮'], axis=1) for town in town_names}
        date_column = forecast_df[forecast_df['站名'] == town_names[0]]['日期']
        forecast_df = pd.concat([date_column] + [forecast_dfs[town].add_suffix(f'_{town}') for town in town_names], axis=1)

    return forecast_df


def read_historical_weather_observation_data(data_fn, start_date=start_date, end_date=end_date, transform_columns=True):
    weather_obs_big_df = pd.read_csv(data_fn)
    return convert_weather_obseravtion_data(weather_obs_big_df, start_date=start_date, end_date=end_date, transform_columns=transform_columns)


def read_historical_power_data(
        data_fn,
        start_date=start_date, end_date=end_date,
        agg_type_list=['max', 'min', 'mean', 'median'],
        agg_feature_list=['風力', '太陽能', '尖峰負載'],
        agg_duration=90
        ):
    power_historical_data_df = pd.read_csv(data_fn)
    power_historical_data_df['日期'] = pd.to_datetime(power_historical_data_df['日期'])

    ## 定義歷史資料的時間範圍
    power_historical_data_df = designate_date_range(power_historical_data_df, '日期', start_date, end_date)

    #將尖峰負載與發電量的單位統一為萬千瓦
    power_historical_data_df['尖峰負載'] /= 10

    ##  將發電機組資料轉換成發電方式資料
    power_type_dict = {
        '日期': power_historical_data_df['日期'],
        '尖峰負載': power_historical_data_df['尖峰負載'],
    }

    for p_type in power_types.power_generation_type.keys():
        power_type_time_series = np.zeros(len(power_historical_data_df),)
        for generator in power_types.power_generation_type[p_type]:
            power_type_time_series += power_historical_data_df[generator]
        power_type_dict[p_type] = list(power_type_time_series)
    power_type_df = DataFrame(power_type_dict)

    big_power_type_dict = {
        '日期': list(power_type_df['日期']),
        '尖峰負載': list(power_type_df['尖峰負載'])
    }

    for big_type in power_types.power_generation_big_type.keys():
        time_series = np.zeros(len(power_historical_data_df),)
        for p_type in power_types.power_generation_big_type[big_type]:
            time_series += power_type_df[p_type]
        big_power_type_dict[big_type] = list(time_series)

    big_power_type_df = DataFrame(big_power_type_dict)
    big_power_type_df['夜尖峰'] = [0 if se > 20 else 1 for se in big_power_type_df['太陽能']]
    big_power_type_df = delete_and_fill_na(big_power_type_df)

    for agg_type in agg_type_list:
        big_power_type_df = get_period_agg_power(big_power_type_df, power_features=agg_feature_list, duration=agg_duration, agg_type=agg_type)

    return big_power_type_df


# 加入日期數字、假日、白天長度等可以由日期決定的資訊
def add_date_related_information(df, daytime_fn):
    # 日期數字化
    first_date = datetime.datetime.strptime(FIRST_DATE_STR, '%Y-%m-%d')
    df['日期數字'] = [(this_date - first_date)/datetime.timedelta(days=1) for this_date in df['日期']]           

    # 加入假日與工作日變量
    df['假日'] = [holiday_identify(d) for d in df['日期']]
    df['週六'] = [1 if d.weekday() == 5 else 0 for d in df['日期']]
    df['週日'] = [1 if d.weekday() == 6 else 0 for d in df['日期']]
    df['補班'] = [1 if d.weekday() in adjusted_work_days else 0 for d in df['日期']]

    # 加入季節變量
    df['1~3月'] = [1 if d.month in [1, 2, 3] and np.sum(df[['假日', '週六', '週日']].iloc[i]) == 0 else 0\
                                for i, d in enumerate(df['日期'])]
    df['11~12月'] = [1 if d.month in [11, 12] and np.sum(df[['假日', '週六', '週日']].iloc[i]) == 0 else 0\
                                for i, d in enumerate(df['日期'])]
    
    # 加入白天長度
    daytime_df = pd.read_csv(daytime_fn)
    daytime_df['日期'] = pd.to_datetime(daytime_df['日期'])
    df = pd.merge(df, daytime_df, on='日期')

    return df


def prepare_forecast_observation_df(historical_data_path: str, start_date: str=start_date, end_date: str=end_date):
    ''' Make combined DataFrame from historical weather forecast and observation data.
    Arg:
    historical_data_path(str): Historical data path.
    start_date(str, optional): Start date of extracted data.
    end_date(str, optional): End date of extracted data.

    Return:
    fore_obs_df: Combined DataFrame
    '''
    start_date = convert_str_to_datetime(start_date)
    end_date = convert_str_to_datetime(end_date)

    # 預報資料
    forecast_df = read_historical_forecast_data(historical_data_path + 'weather/finalized/weather_forecast.csv',
                                                 start_date, end_date)
    forecast_df['站名'] = [town_and_station[town] for town in forecast_df['鄉鎮']]

    # 修改欄位名
    column_mapping = {col: col.replace('_', '預報_') for col in forecast_df.columns if '_' in col}
    forecast_df.rename(column_mapping, axis=1, inplace=True)
    
    # 觀測資料
    observation_df = read_historical_weather_observation_data(historical_data_path + 'weather/finalized/big_table.csv', 
                                                              start_date, end_date, transform_columns=False)
    fore_obs_df = pd.merge(forecast_df, observation_df, on=['站名', '日期'], how='inner')
    return fore_obs_df


def prepare_forecast_power_df(historical_data_path: str, start_date: str=start_date, end_date: str=end_date):
    ''' Make combined DataFrame from historical weather forecast and power data.
    Arg:
    historical_data_path(str): Historical data path.
    start_date(str, optional): Start date of extracted data.
    end_date(str, optional): End date of extracted data.

    Return:
    fore_obs_df: Combined DataFrame
    '''
    start_date = convert_str_to_datetime(start_date)
    end_date = convert_str_to_datetime(end_date)

    # 預報資料
    forecast_df = read_historical_forecast_data(historical_data_path + 'weather/finalized/weather_forecast.csv',
                                                 start_date, end_date,
                                                 transform_columns=True)
    
    # 電力資料
    big_power_type_df = read_historical_power_data(historical_data_path + 'power/power_generation_data.csv',
                                                   start_date, end_date)
    
    # 合併電力與預報兩個DataFrame
    forecast_power_df = pd.merge(big_power_type_df, forecast_df, on='日期', how='inner')

    # 增加日期相關特徵
    forecast_power_df = add_date_related_information(forecast_power_df, historical_data_path + 'daytime/daytime.csv')
    return forecast_power_df


def prepare_observation_power_df(historical_data_path: str, start_date: str=start_date, end_date: str=end_date):
    ''' Make combined DataFrame from historical weather observation and power data.
    Arg:
    historical_data_path(str): Historical data path.
    start_date(str, optional): Start date of extracted data.
    end_date(str, optional): End date of extracted data.

    Return:
    fore_obs_df: Combined DataFrame
    '''
    start_date = convert_str_to_datetime(start_date)
    end_date = convert_str_to_datetime(end_date)
    # 電力資料
    big_power_type_df = read_historical_power_data(historical_data_path + 'power/power_generation_data.csv',
                                                   start_date, end_date)

    # 天氣資料
    weather_df = read_historical_weather_observation_data(historical_data_path + 'weather/finalized/big_table.csv',
                                                          start_date, end_date, transform_columns=True)

    # 合併電力與天氣兩個DataFrame
    weather_power_df = pd.merge(big_power_type_df, weather_df, on='日期', how='inner')

    # 增加日期相關特徵
    weather_power_df = add_date_related_information(weather_power_df, historical_data_path + 'daytime/daytime.csv')

    return weather_power_df


def prepare_data(historical_data_path, start_date=start_date, end_date=end_date):
    return prepare_observation_power_df(historical_data_path=historical_data_path, start_date=start_date, end_date=end_date)
