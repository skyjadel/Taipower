import pandas as pd
import datetime

from crawler import power_generation
from crawler import weather_forecast
from crawler import weather_observation
from today_peak import get_power_generation_at_peak
from save_tree_dfs import save_tree_dfs
from save_RAG_SQL_db import main as save_RAG_SQL

realtime_data_path = '/opt/airflow/sql_db/'
power_structure_path = '/opt/airflow/historical_data/power/power_structure/'

sql_db_path = '/opt/airflow/sql_db/realtime.db'
data_path = '/opt/airflow/sql_db/'
test_db_path = './test.db'

peak_data_path = '/opt/airflow/historical_data/prediction/'
rag_sql_db_fn = power_structure_path + 'RAG_sql.db'

def main_power(sql_db_path=sql_db_path,
               realtime_data_path=realtime_data_path,
               power_structure_path=power_structure_path,
               peak_data_path=peak_data_path):
    # 爬取電力資料，並存成 SQL 資料庫與 csv 檔
    power_generation.get_data(sql_db_path)
    # 存樹狀圖需要的表格
    save_tree_dfs(sql_db_path, power_structure_path)
    # 提取當日尖峰時刻相關資料
    if datetime.datetime.now().hour >= 6:
        peak_dict = get_power_generation_at_peak(sql_db_path)
        pd.DataFrame(peak_dict, index=[0]).to_csv(f'{realtime_data_path}peak.csv', index=False, encoding='utf-8-sig')
    # 更新給機器人查詢的 SQL Database
    save_RAG_SQL(sql_db_fn=rag_sql_db_fn, peak_data_path=peak_data_path, power_data_path=power_structure_path)
    
def main_weather(sql_db_path=sql_db_path):
    # 爬取氣象預報與觀測資料
    weather_forecast.get_data(sql_db_path)
    weather_observation.get_data(sql_db_path)

