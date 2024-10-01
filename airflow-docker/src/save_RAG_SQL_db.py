# 這個檔案負責創建與更新 RAG_SQL.db，以供對話機器人查詢
# 主要被呼叫的函數為 main()

import pandas as pd
import sqlite3
import os
import numpy as np
import datetime

power_data_path = '../../historical/data/power/power_structure/'
peak_data_path = '../../historical/data/prediction/'
sql_db_fn = power_data_path + 'RAG_sql.db'


# 轉換電廠名字，讓機器人容易識別
def arrange_power_plant_name(plant):
    if '大林電廠' in plant:
        return '大林電廠'
    if 'CC' in plant:
        return plant.replace('CC', '')
    if '.' in plant:
        return plant.replace('.', '')
    if 'Gas' in plant:
        return plant.replace('Gas', '電廠')
    if 'GT' in plant:
        return plant.replace('GT', '')
    return plant


# 更新即時與歷史發電結構資料表
def update_power_structure_data(sql_db_fn=sql_db_fn, power_data_path=power_data_path, data_time='many_days'):
    # 依照 date_time 來設定參數
    if data_time == 'one_day':
        table_name = 'real_time_power_generation'
        value_column_name = 'realtime_generation_MW'
        time_column_name = 'time'
        time_column_type = 'TIME'
        data_path = power_data_path + 'today/'
    elif data_time == 'many_days':
        table_name = 'total_daily_power_generation'
        value_column_name = 'daily_generation_GWh'
        time_column_name = 'date'
        time_column_type = 'DATE'
        data_path = power_data_path

    file_list = [data_path + s for s in os.listdir(data_path) if '.csv' in s]

    conn = sqlite3.connect(sql_db_fn)
    cursor = conn.cursor()

    # 如果資料表不存在，就創建資料表
    sql_command = (
        f'CREATE TABLE IF NOT EXISTS {table_name}('
        f'{time_column_name} {time_column_type},'
        'unit VARCHAR(50),'
        'plant VARCHAR(50),'
        'type VARCHAR(50),'
        f'{value_column_name} FLOAT);'
    )
    cursor.execute(sql_command)

    # 在取一天之內不同時間的數據的時候 (datetime='one_day')
    # 如果 csv 檔裡面的最新時間比 SQL 裡面的最新時間要舊，表示 csv 檔已經是新一天的資料
    # 若是如此，就把 SQL 裡面 real_time_power_generation 資料表裡的資料清空，準備存新一天的資料
    if data_time == 'one_day':
        datetime_str_list = []
        for fn in file_list:
            datetime_str = fn.split('/')[-1].split('.')[0].split('_')[-1].replace('-', ':') + ':00'
            datetime_str_list.append(datetime_str)
        
        time_list = [datetime.datetime.strptime(s, '%H:%M:%S').time() for s in datetime_str_list]
        max_csv_time = max(time_list)

        sql_command = 'SELECT MAX(time) FROM real_time_power_generation'
        cursor.execute(sql_command)
        result = cursor.fetchall()
        max_sql_time = datetime.datetime.strptime(result[0][0], '%H:%M:%S').time()

        # 如果 SQL 資料表的最新時間大於 csv 檔的最新時間，則清空 SQL 資料表
        if max_sql_time > max_csv_time:
            sql_command = f'DELETE FROM {table_name}'
            cursor.execute(sql_command)
    
    # Iterate over files in data path, extract data, then insert them into SQL table.
    for i, fn in enumerate(file_list):
        if data_time == 'one_day':
            datetime_str = datetime_str_list[i]
        else:
            datetime_str = fn.split('/')[-1].split('.')[0]

        df = pd.read_csv(fn)

        for i in range(len(df)):
            if df.iloc[i]['depth'] == 3:
                unit = df['id'].iloc[i]
                plant = df['parent'].iloc[i]
                g_type = df['1st_level'].iloc[i].split('(')[1][0:-1]
                value = df['value'].iloc[i]

                plant = arrange_power_plant_name(plant)
                
                # 如果 SQL 裡面有時間與機組重複的資料，則檢查數值，若不同則更新
                # 如果沒有重複的 SQL 資料，則在 SQL 中新增一行
                sql_command = f"SELECT * FROM {table_name} WHERE {time_column_name} = '{datetime_str}' AND unit = '{unit}';"
                cursor.execute(sql_command)
                output = cursor.fetchall()

                if len(output) >= 1:
                    if not output[0][-1] == value:
                        sql_command = f"UPDATE {table_name} SET {value_column_name} = {value} WHERE {time_column_name} = '{datetime_str}' AND unit = '{unit}';"
                    else:
                        sql_command = ''
                else:
                    sql_command = f"INSERT INTO {table_name} VALUES ('{datetime_str}', '{unit}', '{plant}', '{g_type}', {value});"

                cursor.execute(sql_command)
    
    conn.commit()
    cursor.close()
    conn.close()


def update_peak_data(sql_db_fn=sql_db_fn, peak_data_path=peak_data_path, type='prediction'):
    # 依照 type 來設定參數
    if type == 'prediction':
        csv_filename = peak_data_path + 'power.csv'
        table_name = 'peak_load_prediction'
        err_table_command = ''
    elif type == 'truth':
        csv_filename = peak_data_path + 'evaluation.csv'
        table_name = 'peak_load_truth'
        err_table_command = (
            ', wind_energy_error_MW FLOAT, '
            'solar_energy_error_MW FLOAT, '
            'total_error_MW FLOAT'
        )
    
    conn = sqlite3.connect(sql_db_fn)
    cursor = conn.cursor()

    # 如果資料表不存在，就創建資料表
    sql_command = (
        f'CREATE TABLE IF NOT EXISTS {table_name}('
        'date DATE,'
        'wind_energy_MW FLOAT,'
        'solar_energy_MW FLOAT,'
        'total_MW FLOAT'
        f'{err_table_command})'
    )
    cursor.execute(sql_command)

    df = pd.read_csv(csv_filename)

    # 將 DataFrame 當中的每一行存進資料表
    for i in range(len(df)):
        if '/' in df['日期'].iloc[i] or '-' in df['日期'].iloc[i]:
            date_str = df['日期'].iloc[i].replace('/', '-')
            wind_value = float(df['風力'].iloc[i]) * 10
            solar_value = float(df['太陽能'].iloc[i]) * 10
            total_value = float(df['尖峰負載'].iloc[i]) * 10
            if type == 'truth':
                wind_error = np.abs(wind_value - float(df['風力_預測'].iloc[i]) * 10)
                solar_error = np.abs(solar_value - float(df['太陽能_預測'].iloc[i]) * 10)
                total_error = np.abs(total_value - float(df['尖峰負載_預測'].iloc[i]) * 10)

            sql_command = f"SELECT * FROM {table_name} WHERE date = '{date_str}';"
            cursor.execute(sql_command)
            output = cursor.fetchall()

            # 如果 SQL 裡面有日期重複的資料就更新數值
            # 如果沒有重複的 SQL 資料，就在 SQL 中新增一行
            if len(output) >= 1:
                if type == 'prediction':
                    sql_command = (
                        f'UPDATE {table_name} '
                        f"SET wind_energy_MW = {wind_value}, solar_energy_MW = {solar_value}, total_MW = {total_value} "
                        f"WHERE date = '{date_str}';"
                    )

                elif type == 'truth':
                    sql_command = (
                        f'UPDATE {table_name} '
                        f"SET wind_energy_MW = {wind_value}, solar_energy_MW = {solar_value}, total_MW = {total_value}, "
                        f"wind_energy_error_MW = {wind_error}, solar_energy_error_MW = {solar_error}, total_error_MW = {total_error} "
                        f"WHERE date = '{date_str}';"
                    )

            else:
                if type == 'prediction':
                    sql_command = f"INSERT INTO {table_name} VALUES ('{date_str}', {wind_value}, {solar_value}, {total_value});"
                elif type == 'truth':
                    sql_command = f"INSERT INTO {table_name} VALUES ('{date_str}', {wind_value}, {solar_value}, {total_value}, {wind_error}, {solar_error}, {total_error});"

            cursor.execute(sql_command)
    
    conn.commit()
    cursor.close()
    conn.close()


def main(sql_db_fn=sql_db_fn, peak_data_path=peak_data_path, power_data_path=power_data_path):
    # 更新 SQL 資料庫的四張表
    update_power_structure_data(sql_db_fn=sql_db_fn, power_data_path=power_data_path, data_time='many_days')
    update_power_structure_data(sql_db_fn=sql_db_fn, power_data_path=power_data_path, data_time='one_day')
    update_peak_data(sql_db_fn=sql_db_fn, peak_data_path=peak_data_path, type='prediction')
    update_peak_data(sql_db_fn=sql_db_fn, peak_data_path=peak_data_path, type='truth')


if __name__ == '__main__':
    main()

