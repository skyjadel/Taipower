from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/airflow/')
from get_weather_data import main as w_main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'w_dag',
    default_args=default_args,
    description='Run w.py main() every 6 hours',
    schedule_interval='0 */6 * * *',
)

run_w = PythonOperator(
    task_id='run_w_main',
    python_callable=w_main,
    dag=dag,
)