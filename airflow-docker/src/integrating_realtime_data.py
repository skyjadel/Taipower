from data_integration import integrating_forecast, integrating_power_data, integrating_weather_data

test_sql_fn = './../../realtime/realtime_data/realtime.db'
test_hd_path = '../../historical/data/'

def main(sql_db_fn, historical_data_path):
    integrating_forecast.main(sql_db_fn, historical_data_path)
    integrating_power_data.main(sql_db_fn, historical_data_path, solar_energy_day_only=False)
    integrating_weather_data.main(sql_db_fn, historical_data_path)


if __name__ == '__main__':
    print('Start!')
    print(test_hd_path)
    main(test_sql_fn, test_hd_path)