# 這個模組的目的是將 SQL 資料庫中的預報資料整合到歷史預報資料 csv 檔中
# 經由呼叫 main() 完成

import sqlite3
import pandas as pd
import datetime

import sys
sys.path.append('../')

from utils.station_info import town_and_station

strptime = datetime.datetime.strptime
strftime = datetime.datetime.strftime

lastest_forecast_sample_hr = 19 # 整合每天預報數據時，最晚取到這個時間發布的預報

test_sql_fn = '../realtime/realtime_data/realtime.db'
test_hd_path = '../historical/data/'

station_and_town = {v: k for k, v in town_and_station.items()}

wind_direction_dict = {'偏北風':360,
                       '偏南風':180,
                       '偏東風':90,
                       '偏西風':270,
                       '東北風':45,
                       '東南風':135,
                       '西北風':315,
                       '西南風':215}

weather_condition_encoding = {'晴': 0,
                              '多雲': 1,
                              '陰': 2,
                              '短暫陣雨': 3,
                              '短暫陣雨或雷雨': 4,
                              '午後短暫雷陣雨': 5,
                              '陣雨或雷雨': 6}

weather_condition_list = [k for k in weather_condition_encoding.keys()]
town_list = [v for v in station_and_town.values()]

def retrieve_forecast_from_sql(town, sql_db_path, start_date=None, end_date=None):
    conn = sqlite3.connect(sql_db_path)
    cursor = conn.cursor()
    
    sql_command = f"SELECT * FROM forecast WHERE town = '{town}' "
    if not start_date is None:
        date_string = strftime(start_date, '%Y/%m/%d %H:%M:%S')
        sql_command += f"AND forecast_time >= '{date_string}' "
    if not end_date is None:
        date_string = strftime(end_date, '%Y/%m/%d %H:%M:%S')
        sql_command += f"AND forecast_time < '{date_string}' "

    cursor.execute(sql_command)
    rows = cursor.fetchall()

    forecast_dict = {'鄉鎮': [],
                     '預測時間':[],
                     '目標時間':[],
                     '天氣狀況':[],
                     '溫度':[],
                     '降水機率':[],
                     '相對溼度':[],
                     '風速':[],
                     '風向':[]}
    
    for row in rows:
        forecast_dict['鄉鎮'].append(row[0])
        forecast_dict['預測時間'].append(strptime(row[1], '%Y/%m/%d %H:%M:%S'))
        forecast_dict['目標時間'].append(strptime(row[2], '%Y/%m/%d %H:%M:%S'))
        forecast_dict['天氣狀況'].append(row[3])
        forecast_dict['溫度'].append(row[4])
        forecast_dict['降水機率'].append(row[5])
        forecast_dict['相對溼度'].append(row[6])
        forecast_dict['風速'].append(row[7])
        forecast_dict['風向'].append(row[8])
        
    cursor.close()
    conn.close()
    return pd.DataFrame(forecast_dict)

def retrieve_update_times_from_sql(sql_db_path):
    conn = sqlite3.connect(sql_db_path)
    cursor = conn.cursor()
    
    sql_command = 'SELECT update_time FROM forecast'
    cursor.execute(sql_command)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    update_strs = list(set([r[0].split(':')[0] for r in rows]))

    update_times = {}
    for row in update_strs:
        d = row.split(' ')[0]
        t = int(row.split(' ')[1])
        if not d in update_times.keys():
            update_times[d] = [t]
        else:
            update_times[d].append(t)

    for k in update_times.keys():
        update_times[k].sort()
    return update_times

def convert_to_timestamp(input_time):
    if str(type(input_time)) ==  "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
        return input_time
    if str(type(input_time)) == "<class 'datetime.datetime'>":
        return pd.Timestamp(input_time)
    if type(input_time) == str:
        return pd.Timestamp(strptime(input_time, '%Y/%m/%d %H:%M:%S'))
    
def sample_forecast_at_given_time(forecast_df, sample_time):
    sample_time = convert_to_timestamp(sample_time)

    forecast_df = forecast_df[forecast_df['預測時間'] < sample_time]
    
    update_times = list(set(forecast_df['預測時間']))
    update_times.sort()
    if len(update_times) > 0:
        newest_time = update_times[-1]
    else:
        return None

    forecast_df = forecast_df[forecast_df['預測時間'] == newest_time]
    return forecast_df

def sample_forecast_with_given_deltaday(forecast_df, deltaday=1):
    delta_timestamp = pd.Timedelta(days=deltaday)

    forecast_df['day_diff'] = [forecast_df['目標時間'].iloc[i].date() - forecast_df['預測時間'].iloc[i].date() for i in range(len(forecast_df))]
    forecast_df = forecast_df[forecast_df['day_diff'] == delta_timestamp]
    forecast_df = forecast_df.copy()
    forecast_df.drop('day_diff', axis=1, inplace=True)
    forecast_df.reset_index(drop=True, inplace=True)
    return forecast_df

def encode_oneday_forecast_data(forecast_df):
    this_town = forecast_df['鄉鎮'].iloc[0]
    this_date = forecast_df['目標時間'].iloc[0].date()
    this_df = forecast_df.drop('預測時間', axis=1)
    this_df['Time'] = [t.hour for t in this_df['目標時間']]

    new_dict = {'鄉鎮': this_town, '日期': this_date}
    for h in range(0, 24, 3):
        this_row = this_df[this_df['Time']==h]
        for col in this_row.columns:
            if not col in ['鄉鎮', '目標時間', 'Time']:
                if col == '天氣狀況':
                    for con in weather_condition_list:
                        this_key = f'{con}_{h}'
                        new_dict[this_key] = int(con == this_row[col].iloc[0])
                elif col == '風向':
                    this_key = f'{col}_{h}'
                    new_dict[this_key] = wind_direction_dict[this_row[col].iloc[0]]
                else:
                    this_key = f'{col}_{h}'
                    new_dict[this_key] = this_row[col].iloc[0]
    return pd.DataFrame([new_dict])

def arrange_forecast_for_given_town(town, sql_db_path, forecast_times, historical_df=None,
                                    sample_hr=lastest_forecast_sample_hr, least_integrate_days=5):
    
    if not historical_df is None:
        this_historical_df = historical_df[historical_df['鄉鎮']==town]
        historical_date_list = list(this_historical_df['日期'])
        historical_date_list.sort()
        latest_historical_date = datetime.datetime.strptime(historical_date_list[-1], '%Y-%m-%d')

        sql_date_str_list = list(forecast_times.keys())
        sql_date_str_list.sort()
        latest_sql_date_str = sql_date_str_list[-1]
        start_date = datetime.datetime.strptime(latest_sql_date_str, '%Y/%m/%d') - datetime.timedelta(days=least_integrate_days-1)

        start_date = min(start_date, latest_historical_date + datetime.timedelta(days=1)).date()

        forecast_date_list = []
        for d in forecast_times.keys():
            if datetime.datetime.strptime(d, '%Y/%m/%d').date() >= start_date:
                forecast_date_list.append(d)
        forecast_times = {d:forecast_times[d] for d in forecast_date_list}
    else:
        start_date = None

    init_df = retrieve_forecast_from_sql(town, sql_db_path=sql_db_path, start_date=start_date)
    ran_dates = []
    df_list = []
    if not historical_df is None:
        #this_historical_df['日期'] = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in this_historical_df['日期']]
        df_list.append(this_historical_df)

    for d, ts in forecast_times.items():
        if (not d in ran_dates) and sample_hr > ts[0]:
            this_df = sample_forecast_at_given_time(init_df, f'{d} {sample_hr}:00:00')
            if this_df is None:
                continue
            this_df = sample_forecast_with_given_deltaday(this_df)
            this_df = encode_oneday_forecast_data(this_df)

            this_df['日期'] = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in this_df['日期']]

            ran_dates.append(d)
            df_list.append(this_df)
    if len(df_list) == 0:
        return None
    
    return_df = pd.concat(df_list, axis=0, ignore_index=True)
    return_df = return_df.drop_duplicates(['日期', '鄉鎮'], keep='last').sort_values('日期').reset_index(drop=True)
    return return_df

def arrange_forecast_for_towns(towns, sql_db_path, forecast_times, historical_df=None,
                               sample_hr=lastest_forecast_sample_hr, least_integrate_days=5):
    df_list = []
    for town in towns:
        this_df = arrange_forecast_for_given_town(town, sql_db_path, historical_df=historical_df,
                                                  sample_hr=sample_hr, forecast_times=forecast_times,
                                                  least_integrate_days=least_integrate_days)
        if not this_df is None:
            df_list.append(this_df)
    return pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)


def main(sql_db_fn, historical_data_path, save_file=True, least_integrate_days=5):
    # 將 SQL 資料庫中的預報資料整合到歷史預報資料 csv 檔中
    forecast_times = retrieve_update_times_from_sql(sql_db_fn)
    historical_df = pd.read_csv(historical_data_path+'weather/finalized/weather_forecast.csv')
    #print(forecast_times)
    df = arrange_forecast_for_towns(town_list, sql_db_fn, historical_df=historical_df,
                                    forecast_times=forecast_times, least_integrate_days=least_integrate_days)
    
    if df is None:
        return None
    
    if save_file:
        df.to_csv(historical_data_path+'weather/finalized/weather_forecast.csv', encoding='utf-8-sig', index=False)
        return None
    


if __name__ == '__main__':
    print('Start!')
    print(test_hd_path)
    main(test_sql_fn, test_hd_path, save_file=False)


