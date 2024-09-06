import sqlite3
import pandas as pd
import datetime
import numpy as np

from utils.sun_light import calculate_daytime, calculate_all_day_sunlight
from utils.station_info import site_location_dict

test_sql_fn = './../../realtime/realtime_data/realtime.db'
test_hd_path = '../../historical copy/data/'


def sun_light_time_to_energy(hr):
    return hr * 2.5198 + 8.6888

wind_direction_dict = {'北':360,
                       '南':180,
                       '東':90,
                       '西':270,
                       '東北':45,
                       '東南':135,
                       '西北':315,
                       '西南':215,
                       '北北東':22.5,
                       '東北東':67.5,
                       '東南東':112.5,
                       '南南東':157.5,
                       '南南西':202.5,
                       '西南西':237.5,
                       '西北西':292.5,
                       '北北西':337.5}

def get_avg_wind_direction(wind_speed_list, wind_direction_list):
    EW_wind = []
    NS_wind = []
    for i in range(len(wind_speed_list)):
        try:
            this_wind_speed = float(wind_speed_list[i])
            if np.isnan(this_wind_speed):
                this_wind_speed = None
        except:
            this_wind_speed = None
        
            
        try:
            this_angle = float(wind_direction_list[i]) / 180 * np.pi
        except:
            if wind_direction_list[i] in wind_direction_dict.keys():
                this_angle = wind_direction_dict[wind_direction_list[i]] / 180 * np.pi
                if np.isnan(this_angle):
                    this_wind_speed = None
            else:
                this_angle = None
        
        if not (this_angle is None or this_wind_speed is None):
            EW_wind.append(this_wind_speed * np.sin(this_angle))
            NS_wind.append(this_wind_speed * np.cos(this_angle))

    if len(EW_wind) == 0:
        return None
    direction = int(round((np.angle(np.mean(EW_wind) * 1j + np.mean(NS_wind)) / np.pi * 180) % 360, ndigits=-1))
    if direction == 0:
        direction = 360
    return direction


def get_rainfall_hr(data_list):
    hr_rate = 0.5
    rainfall_hr = 0
    try:
        previous_rf = float(data_list[0])
        if np.isnan(previous_rf):
            previous_rf = None
    except:
        previous_rf = None

    time_gap = 1
    valid_count = 0
    for i in range(1, len(data_list)):
        try:
            this_rf = float(data_list[i])
            if np.isnan(this_rf):
                this_rf = None
        except:
            this_rf = None
            
        if not (previous_rf is None or this_rf is None):
            valid_count += 1
            if this_rf > previous_rf:
                rainfall_hr += time_gap
            time_gap = 1
            previous_rf = this_rf
        elif not this_rf is None:
            previous_rf = this_rf
        else:
            time_gap += 1
    if valid_count == 0:
        return None
    return rainfall_hr * hr_rate

def nanmean(L):
    if np.prod(np.array(L).shape) == 0:
        return None
    try:
        return np.nanmean(L)
    except:
        return None
    

def get_oneday_weather_observation_data(date, station, sql_db_fn):
    target_col_list = ['站名', '日期', '氣溫(℃)', '最高氣溫(℃)', '最低氣溫(℃)', '相對溼度(%)', '風速(m/s)', 
                       '風向(360degree)', '最大瞬間風(m/s)', '最大瞬間風風向(360degree)', '降水量(mm)',
                       '降水時數(hour)', '日照時數(hour)', '日照率(%)', '全天空日射量(MJ/㎡)', '總雲量(0~10)']
    sql_col_list = ['Station', 'Time', 'Temperature', 'Weather', 'Wind_Direction', 'Wind_Speed', 'Gust_Wind', 'Humidity', 'Pressure', 'Rainfall', 'Sunlight']
    
    date_str = datetime.datetime.strftime(date, '%Y/%m/%d')
    date_str_2nd_day = datetime.datetime.strftime(date + datetime.timedelta(days=1), '%Y/%m/%d')
    
    conn = sqlite3.connect(sql_db_fn)
    cursor = conn.cursor()
    
    sql_command = f"SELECT * FROM observation WHERE obs_time > '{date_str} 00:00:00' AND obs_time <= '{date_str_2nd_day} 00:00:00' AND station = '{station}'"
    cursor.execute(sql_command)
    
    sql_output = cursor.fetchall()
    cursor.close()
    conn.close()

    sql_df = pd.DataFrame(sql_output, columns=sql_col_list).sort_values('Time').reset_index(drop=True)

    if len(sql_df) == 0:
        output_dict = {k:np.nan for k in target_col_list}
        output_dict['站名'] = [station]
        output_dict['日期'] = [date_str]
        return output_dict, sql_df
    output_dict = {}
    output_dict['站名'] = [station]
    output_dict['日期'] = [date_str]
    output_dict['氣溫(℃)'] = [nanmean(sql_df['Temperature']) if nanmean(sql_df['Temperature']) is None else round(nanmean(sql_df['Temperature']), ndigits=1)]
    output_dict['最高氣溫(℃)'] = [np.max(sql_df['Temperature'])]
    output_dict['最低氣溫(℃)'] = [np.min(sql_df['Temperature'])]
    output_dict['相對溼度(%)'] = [nanmean(sql_df['Humidity']) if nanmean(sql_df['Temperature']) is None else round(nanmean(sql_df['Humidity']))]
    output_dict['風速(m/s)'] = [nanmean(sql_df['Wind_Speed']) if nanmean(sql_df['Temperature']) is None else round(nanmean(sql_df['Wind_Speed']), ndigits=1)]
    output_dict['風向(360degree)'] = [get_avg_wind_direction(sql_df['Wind_Speed'], sql_df['Wind_Direction'])]
    output_dict['最大瞬間風(m/s)'] = [np.max(sql_df['Gust_Wind'])]
    row_of_max_gust_wind = sql_df[sql_df['Gust_Wind'] == np.max(sql_df['Gust_Wind'])]
    output_dict['最大瞬間風風向(360degree)'] = [get_avg_wind_direction(list(row_of_max_gust_wind['Gust_Wind']), list(row_of_max_gust_wind['Wind_Direction']))]
    output_dict['降水量(mm)'] = [sql_df['Rainfall'].iloc[-1]]
    output_dict['降水時數(hour)'] = [get_rainfall_hr(sql_df['Rainfall'])]
    output_dict['日照時數(hour)'] = [sql_df['Sunlight'].iloc[-1]]
    if output_dict['日照時數(hour)'][0] is None:
        output_dict['日照率(%)'] = output_dict['日照時數(hour)']
        output_dict['全天空日射量(MJ/㎡)'] = output_dict['日照時數(hour)']
        output_dict['總雲量(0~10)'] = output_dict['日照時數(hour)']
    else:
        output_dict['日照率(%)'] = [round(sql_df['Sunlight'].iloc[-1] / calculate_daytime(site_location_dict[station], date_str.replace('/', '-')) * 100, ndigits=1)]
        eff_sunlight_hr = calculate_all_day_sunlight(site_location_dict[station], date_str.replace('/', '-')) * output_dict['日照率(%)'][0] / 100
        output_dict['全天空日射量(MJ/㎡)'] = [round(sun_light_time_to_energy(eff_sunlight_hr), ndigits=2)]
        output_dict['總雲量(0~10)'] = [round((100 - output_dict['日照率(%)'][0]) / 10, ndigits=1)]

    for k, v in output_dict.items():
        if not (v[0] is None or type(v[0]) == str):
            if np.isnan(v[0]):
                output_dict[k] = [None]
    
    return output_dict, sql_df

def main(sql_db_fn, historical_data_path, time_zone='TWN'):
    historical_weather_df = pd.read_csv(historical_data_path + 'weather/finalized/big_table.csv')

    time_now = datetime.datetime.now()
    date_yesterday = time_now.date() - datetime.timedelta(days=1)
    if time_now.hour < 1 and time_now.minute <= 15:
        date_yesterday -= datetime.timedelta(days=1)

    station_list = site_location_dict.keys()
    new_data = []
    for station in station_list:
        this_station_df = historical_weather_df[historical_weather_df['站名']==station]
        new_data.append(this_station_df)
        latest_date_in_historical_weather_data = datetime.datetime.strptime(max(this_station_df['日期']), '%Y/%m/%d').date()
        delta_days = (date_yesterday - latest_date_in_historical_weather_data).days
        for d in range(1, delta_days+1):
            output_dict, _ = get_oneday_weather_observation_data(latest_date_in_historical_weather_data + datetime.timedelta(days=d), station, sql_db_fn)
            new_data.append(pd.DataFrame(output_dict))
    new_df = pd.concat(new_data, axis=0, ignore_index=True).reset_index(drop=True)

    new_df.to_csv(historical_data_path + 'weather/finalized/big_table.csv', encoding='utf-8-sig', index=False)

if __name__ == '__main__':
    print('Start!')
    print(test_hd_path)
    main(test_hd_path, test_sql_fn)
