# 從中央氣象署網站抓取即時氣象預報資料
# 呼叫 get_data() 完成

import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import sqlite3
import datetime

from utils.station_info import town_id_table


def get_weather_forecast(town_name):
    county_id, town_id = town_id_table[town_name]
    
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    headers = {"User-Agent": user_agent}
    req = requests.get(url=f'https://www.cwa.gov.tw/Data/js/3hr/ChartData_3hr_T_{county_id}.js', headers=headers)
    req.encoding = 'utf-8'
    js_str = req.text

    #取得預報發布時間
    forecast_time_str = js_str[js_str.index('Updated:')+9:js_str.index('Updated:')+28]
    forecast_time = datetime.datetime.strptime(forecast_time_str, '%Y/%m/%d %H:%M:%S')

    #取得預報目標時間
    first_forecast_hr_str = js_str[js_str.index("Time_3hr")+18:js_str.index("Time_3hr")+26]
    forecasted_hrs = [datetime.datetime.strptime(f"{forecast_time_str.split('/')[0]}/{first_forecast_hr_str.split(' ')[1]} {first_forecast_hr_str.split(' ')[0]}:00:00",
                      '%Y/%m/%d %H:%M:%S')]
    for i in range(24):
        forecasted_hrs.append(forecasted_hrs[-1] + datetime.timedelta(hours=3))
    
    six_hr = forecasted_hrs[0].hour % 6 == 0

    #抓取預報資料
    time_now = datetime.datetime.now()
    yr_now = time_now.year
    month_now = time_now.month
    day_now = time_now.day
    
    req = requests.get(url=f'https://www.cwa.gov.tw/V8/C/W/Town/MOD/3hr/{town_id}_3hr_PC.html?T={yr_now * 1000000 + month_now * 10000 + day_now * 100}')
    req.encoding = 'utf-8'
    data_str = req.text
    soup = bs(data_str, 'html.parser')

    data = {
        '鄉鎮市區': [town_name] * len(forecasted_hrs),
        '預測發布時間': [forecast_time] * len(forecasted_hrs),
        '預測時間': forecasted_hrs,
        '天氣狀況': [],
        '溫度': [],
        '降雨機率': [],
        '相對濕度': [],
        '風速': [],
        '風向': []
    }
    #組織預報資料
    table_rows = soup.find_all('tr')

    this_row = table_rows[2]
    for td in this_row.find_all('td'):
        data['天氣狀況'].append(td.find('img').attrs['title'])
    
    this_row = table_rows[3]
    for td in this_row.find_all('td'):
        data['溫度'].append(int(td.find('span').text))
    
    this_row = table_rows[5]
    for i, td in enumerate(this_row.find_all('td')):
        this_pop = int(td.text[0:-1]) / 100
        data['降雨機率'].append(this_pop)
        if six_hr or (not i == 0):
            data['降雨機率'].append(this_pop)
        data['降雨機率'] = data['降雨機率'][0:len(forecasted_hrs)]
    
    this_row = table_rows[6]
    for td in this_row.find_all('td'):
        data['相對濕度'].append(int(td.text[0:-1]))
    
    this_row = table_rows[8]
    for td in this_row.find_all('td'):
        data['風速'].append(int(td.text.strip('≥')))
    
    this_row = table_rows[9]
    for td in this_row.find_all('td'):
        data['風向'].append(td.text)
        
    return pd.DataFrame(data)

# 執行 get_weather_forecast 並存入 SQL database
def get_data(sql_db_path):
    for i, town_name in enumerate(town_id_table.keys()):
        if i == 0:
            forecast_df = get_weather_forecast(town_name)
        else:
            forecast_df = pd.concat([forecast_df, get_weather_forecast(town_name)], axis=0, ignore_index=True)
    
    conn = sqlite3.connect(sql_db_path)
    cursor = conn.cursor()

    sql_command = (
        'CREATE TABLE IF NOT EXISTS forecast('
        'town VARCHAR(15), '
        'update_time DATETIME, '
        'forecast_time DATETIME, '
        'weather_condition VARCHAR(10), '
        'temperature INT, '
        'prob_of_precipitation FLOAT, '
        'relative humidity INT, '
        'wind_speed INT, '
        'wind_direction CHAR(5)'
        ');'
    )
    cursor.execute(sql_command)
    conn.commit()

    earliest_update_time = min(forecast_df['預測發布時間'])
    earliest_update_time_str = datetime.datetime.strftime(earliest_update_time, '%Y/%m/%d %H:%M:%S')

    cursor.execute(f"SELECT town, update_time, forecast_time FROM forecast WHERE update_time >= '{earliest_update_time_str}'")
    existing_data = cursor.fetchall()

    for i in range(len(forecast_df)):
        this_list = list(forecast_df.loc[i])
        this_town = this_list[0]
        update_time_str = datetime.datetime.strftime(this_list[1], '%Y/%m/%d %H:%M:%S')
        forecast_time_str = datetime.datetime.strftime(this_list[2], '%Y/%m/%d %H:%M:%S')
        if not (this_town, update_time_str, forecast_time_str) in existing_data:
            sql_command = (
                'INSERT INTO forecast VALUES('
                f"'{this_town}', "
                f"'{update_time_str}', "
                f"'{forecast_time_str}', "
                f"'{this_list[3]}', "
                f"{this_list[4]}, "
                f"{this_list[5]}, "
                f"{this_list[6]}, "
                f"{this_list[7]}, "
                f"'{this_list[8]}'"
                ');'
            )
        conn.execute(sql_command)
    conn.commit()
    cursor.close()
    conn.close()