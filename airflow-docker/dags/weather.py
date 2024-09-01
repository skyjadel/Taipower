from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from realtime_data_download import main_weather as w_main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': startdate,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'w_dag',
    default_args=default_args,
    description='Run w.py main() every 6 hours',
    schedule_interval='5 1,7,13,19 * * *',
)

run_w = PythonOperator(
    task_id='run_w_main',
    python_callable=w_main,
    dag=dag,
)