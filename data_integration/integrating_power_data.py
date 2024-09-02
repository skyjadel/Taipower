import sqlite3
import pandas as pd
import datetime
import numpy as np

#sql_db_fn = './realtime/realtime_data/realtime.db'
#historical_data_path = './historical/data/'
test_sql_fn = './../../realtime/realtime_data/realtime.db'
test_hd_path = '../../historical copy/data/'

generator_translation_dict = {
    '興達': ['興達CC'],
    '通霄': ['通霄CC', '通霄GT'],
    '大潭': ['大潭CC'],
    '南部': ['南部CC'],
    '大林': ['大林#5', '大林#6'],
    '氣渦輪': ['Gas'],
    '離島': ['澎湖', '金門', '馬祖', '離島'],
    '大觀': ['大觀一'],
    '大觀二': ['大觀二'],
    '明潭': ['明潭'],
    '海湖': ['海湖'],
    '豐德': ['豐德'],
    '其他小水力': ['小水力'],
    '協和': ['協和'],
}

hydro_powers = ['德基', '青山', '谷關', '天輪', '馬鞍', '萬大', '鉅工', 
                '立霧', '龍澗', '卓蘭', '水里']
generator_translation_dict.update({k: [k] for k in hydro_powers})

def get_full_oneday_power_df(sql_db_fn, date, day_only):
    #date_num = date.year * 10000 + date.month * 100 + date.day
    date_str = datetime.datetime.strftime(date, '%Y-%m-%d')
    date_str_2nd_day = datetime.datetime.strftime(date + datetime.timedelta(days=1), '%Y-%m-%d')
    conn = sqlite3.connect(sql_db_fn)
    cursor = conn.cursor()
    
    sql_command = f"SELECT * FROM power WHERE time > '{date_str} 00:00' AND time <= '{date_str_2nd_day} 00:00'"
    cursor.execute(sql_command)
    
    sql_output = cursor.fetchall()
    cursor.close()
    conn.close()
    if len(sql_output) < 20:
        return None
    
    sql_df = pd.DataFrame(sql_output, columns=['時間', '分類', '機組', '容量', '發電量'])
    sql_df = sql_df.drop_duplicates(['時間', '分類', '機組'])

    pwd_gen_df = pd.pivot_table(sql_df, index='時間', columns=['機組','分類'], values='發電量')

    if day_only:
        pwd_gen_df['白日'] = [1 if np.abs(11.5 - int(d.split(' ')[1].split(':')[0])) <= 4.6 else 0 for d in pwd_gen_df.index]
        pwd_gen_df = pwd_gen_df[pwd_gen_df['白日']==1]
        pwd_gen_df = pwd_gen_df.drop('白日', axis=1)

    pwd_gen_df['總負載'] = np.sum(pwd_gen_df, axis=1)
    pwd_gen_df.reset_index(inplace=True)
    pwd_gen_df.rename({'Index':'時間'}, inplace=True)

    this_gen_list = [col for col in pwd_gen_df.columns if col[1] == '太陽能(Solar)']
    pwd_gen_df.loc[pwd_gen_df.index, '太陽能發電'] = np.nansum(np.array(pwd_gen_df[this_gen_list]), axis=1)
    pwd_gen_df = pwd_gen_df.drop(this_gen_list, axis=1)

    this_gen_list = [col for col in pwd_gen_df.columns if col[1] == '風力(Wind)']
    pwd_gen_df.loc[pwd_gen_df.index, '風力發電'] = np.nansum(np.array(pwd_gen_df[this_gen_list]), axis=1)
    pwd_gen_df = pwd_gen_df.drop(this_gen_list, axis=1)

    if ('電池', '儲能負載(Energy Storage Load)') in pwd_gen_df.columns:
        pwd_gen_df[('電池', '')] = pwd_gen_df[('電池', '儲能(Energy Storage System)')] + pwd_gen_df[('電池', '儲能負載(Energy Storage Load)')]
        pwd_gen_df.drop([('電池', '儲能(Energy Storage System)'), ('電池', '儲能負載(Energy Storage Load)')], axis=1, inplace=True)

    pwd_gen_df.columns = [col[0] for col in pwd_gen_df.columns]

    for key, v in generator_translation_dict.items():
        this_col_list = []
        for col in pwd_gen_df.columns:
            for this_indicator in v:
                if this_indicator in col:
                    this_col_list.append(col)
                    break
        pwd_gen_df.loc[pwd_gen_df.index, key] = np.nansum(np.array(pwd_gen_df[this_col_list]), axis=1)
        pwd_gen_df = pwd_gen_df.drop(this_col_list, axis=1)
    time_list = list(pwd_gen_df['時間'])
    pwd_gen_df = pwd_gen_df.drop('時間', axis=1) / 10
    pwd_gen_df['時間'] = time_list
    pwd_gen_df['日期'] = date_str.replace('-', '/')
    pwd_gen_df.rename({'總負載': '尖峰負載'}, axis=1, inplace=True)
    pwd_gen_df['尖峰負載'] *= 10
        
    return pwd_gen_df

def get_oneday_power_data(sql_db_fn, date, solar_energy_day_only):
    pwd_gen_df = get_full_oneday_power_df(sql_db_fn, date, day_only=False)
    if pwd_gen_df is None:
        return None
    pwd_gen_df = pwd_gen_df[pwd_gen_df['尖峰負載'] == max(pwd_gen_df['尖峰負載'])]

    if solar_energy_day_only:
        solar_df = get_full_oneday_power_df(sql_db_fn, date, day_only=True)
        if not solar_df is None:
            solar_df = solar_df[solar_df['尖峰負載'] == max(solar_df['尖峰負載'])]
            pwd_gen_df.loc[pwd_gen_df.index[0], '太陽能發電'] = solar_df['太陽能發電'].iloc[0]
    return pwd_gen_df

def load_power_data(sql_db_fn, latest_date_in_historical_power_data, solar_energy_day_only):
    time_now = datetime.datetime.now()
    date_yesterday = time_now.date() - datetime.timedelta(days=1)
    if time_now.hour < 1 and time_now.minute <= 15:
        date_yesterday -= datetime.timedelta(days=1)
            
    delta_days = (date_yesterday - latest_date_in_historical_power_data).days

    new_data = []
    for d in range(1, delta_days+1):
        output_df = get_oneday_power_data(sql_db_fn, latest_date_in_historical_power_data + datetime.timedelta(days=d), solar_energy_day_only)
        if not output_df is None:
            new_data.append(pd.DataFrame(output_df))
    if len(new_data) > 0:
        new_data = pd.concat(new_data, axis=0, ignore_index=True).reset_index(drop=True)
    else:
        new_data = None
    return new_data

def main(sql_db_fn, historical_data_path, solar_energy_day_only):
    historical_df = pd.read_csv(historical_data_path + 'power/power_generation_data.csv')
    historical_dates = pd.to_datetime(historical_df['日期'])
    latest_date_in_historical_power_data = max(historical_dates).to_pydatetime().date()
    new_data = load_power_data(sql_db_fn, latest_date_in_historical_power_data, solar_energy_day_only)

    if new_data is None:
        final_df = historical_df
    else:
        final_df = pd.concat([historical_df ,new_data], axis=0, ignore_index=True).reset_index(drop=True)
        cols_drop = list(set(final_df.columns).difference(set(historical_df.columns)))
        final_df.drop(cols_drop, axis=1, inplace=True)
        final_df = final_df.fillna(0)

    final_df.to_csv(historical_data_path + 'power/power_generation_data.csv', encoding='utf-8-sig', index=False)

if __name__ == '__main__':
    print('Start!')
    print(test_hd_path)
    main(test_sql_fn, test_hd_path, solar_energy_day_only=True)