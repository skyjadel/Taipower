from crawler import power_generation

sql_db_path = '/opt/airflow/sql_db/realtime.db'

def main():
    power_generation.get_data(sql_db_path)

# if __name__ == '__main__':
#     main()