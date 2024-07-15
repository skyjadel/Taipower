from crawler import weather_forecast
from crawler import weather_observation

sql_db_path = '/opt/airflow/sql_db/realtime.db'

def main():
    weather_forecast.get_data(sql_db_path)
    weather_observation.get_data(sql_db_path)

# if __name__ == '__main__':
#     print('Start!')
#     main()