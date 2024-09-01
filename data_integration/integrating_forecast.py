import sqlite3
import pandas as pd
import datetime
strptime = datetime.datetime.strptime
strftime = datetime.datetime.strftime

#sql_db_path = './realtime/realtime_data/realtime.db'
test_sql_fn = './../../realtime/realtime_data/realtime.db'
test_hd_path = '../../historical copy/data/'

station_and_town = {
    '臺北': '臺北市中正區',
    '高雄': '高雄市楠梓區',
    '嘉義': '嘉義市西區',
    '東吉島': '澎湖縣望安鄉',
    '臺中電廠': '臺中市龍井區',
    '臺西': '雲林縣臺西鄉'
}

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
        sql_command += f"AND obs_time >= '{date_string}' "
    if not end_date is None:
        date_string = strftime(end_date, '%Y/%m/%d %H:%M:%S')
        sql_command += f"AND obs_time < '{date_string}' "

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
    newest_time = update_times[-1]

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

def arrange_forecast_for_given_town(town, sql_db_path, forecast_times, sample_hr=15):
    init_df = retrieve_forecast_from_sql(town, sql_db_path=sql_db_path)
    ran_dates = []
    df_list = []
    for d, ts in forecast_times.items():
        if (not d in ran_dates) and sample_hr > ts[0]:
            this_df = sample_forecast_at_given_time(init_df, f'{d} {sample_hr}:00:00')
            this_df = sample_forecast_with_given_deltaday(this_df)
            this_df = encode_oneday_forecast_data(this_df)

            ran_dates.append(d)
            df_list.append(this_df)

    return pd.concat(df_list, axis=0, ignore_index=True).sort_values('日期').reset_index(drop=True)

def arrange_forecast_for_towns(towns, sql_db_path, forecast_times, sample_hr=15):
    df_list = []
    for town in towns:
        df_list.append(arrange_forecast_for_given_town(town, sql_db_path, sample_hr=sample_hr, forecast_times=forecast_times))
    return pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)


def main(sql_db_fn, historical_data_path):
    forecast_times = retrieve_update_times_from_sql(sql_db_fn)
    df = arrange_forecast_for_towns(town_list, sql_db_fn, forecast_times=forecast_times)
    df.to_csv(historical_data_path+'weather/finalized/weather_forecast.csv', encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    print('Start!')
    print(test_hd_path)
    main(test_sql_fn, test_hd_path)

