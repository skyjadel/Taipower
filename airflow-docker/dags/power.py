from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from realtime_data_download import main_power as p_main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': startdate,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'p_dag',
    default_args=default_args,
    description='Run p.py main() every 30 minutes',
    schedule_interval='*/30 * * * *',
)

run_p = PythonOperator(
    task_id='run_p_main',
    python_callable=p_main,
    dag=dag,
)