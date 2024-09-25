import pandas as pd
import sqlite3
import os
import numpy as np

power_data_path = '../../historical/data/power/power_structure/'
peak_data_path = '../../historical/data/prediction/'
sql_db_fn = power_data_path + 'RAG_sql.db'


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


def update_power_structure_data(sql_db_fn=sql_db_fn, power_data_path=power_data_path, data_time='many_days'):
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

    if data_time == 'one_day':
        sql_command = f'DROP TABLE IF EXISTS {table_name}'
        cursor.execute(sql_command)

    sql_command = (
        f'CREATE TABLE IF NOT EXISTS {table_name}('
        f'{time_column_name} {time_column_type},'
        'unit VARCHAR(50),'
        'plant VARCHAR(50),'
        'type VARCHAR(50),'
        f'{value_column_name} FLOAT);'
    )
    cursor.execute(sql_command)

    for fn in file_list:
        if '_' in fn.split('/')[-1]:
            datetime_str = fn.split('/')[-1].split('.')[0].split('_')[-1].replace('-', ':') + ':00'
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
    update_power_structure_data(sql_db_fn=sql_db_fn, power_data_path=power_data_path, data_time='many_days')
    update_power_structure_data(sql_db_fn=sql_db_fn, power_data_path=power_data_path, data_time='one_day')
    update_peak_data(sql_db_fn=sql_db_fn, peak_data_path=peak_data_path, type='prediction')
    update_peak_data(sql_db_fn=sql_db_fn, peak_data_path=peak_data_path, type='truth')


if __name__ == '__main__':
    main()

