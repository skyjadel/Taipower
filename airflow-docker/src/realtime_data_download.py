import pandas as pd

from crawler import power_generation
from crawler import weather_forecast
from crawler import weather_observation
from today_peak import get_power_generation_at_peak
from save_tree_dfs import save_tree_dfs

realtime_data_path = '/opt/airflow/sql_db/'
power_structure_path = '/opt/airflow/historical_data/power/power_structure/'

sql_db_path = '/opt/airflow/sql_db/realtime.db'
data_path = '/opt/airflow/sql_db/'
test_db_path = './test.db'

def main_power(sql_db_path=sql_db_path, data_path=data_path, power_structure_path=power_structure_path):
    # 爬取電力資料，並存成 SQL 資料庫與 csv 檔
    power_generation.get_data(sql_db_path)
    save_tree_dfs(sql_db_path, power_structure_path)
    peak_dict = get_power_generation_at_peak(sql_db_path)
    pd.DataFrame(peak_dict, index=[0]).to_csv(f'{data_path}peak.csv', index=False, encoding='utf-8-sig')
    
def main_weather(sql_db_path=sql_db_path):
    # 爬取氣象預報與觀測資料
    weather_forecast.get_data(sql_db_path)
    weather_observation.get_data(sql_db_path)

if __name__ == '__main__':
    print('Start!')
    print(test_db_path)
    main_power(test_db_path)
    main_weather(test_db_path)