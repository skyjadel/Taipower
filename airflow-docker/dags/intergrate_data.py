from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from integrating_realtime_data import main

sql_db_fn = '/opt/airflow/sql_db/realtime.db'
historical_data_path = '/opt/airflow/historical_data/'

def run():
    main(sql_db_fn, historical_data_path)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': startdate,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'i_dag',
    default_args=default_args,
    description='Run integrating_realtime_data.main() everyday',
    schedule_interval='20 0,19 * * *',
)

run_p = PythonOperator(
    task_id='run_i_main',
    python_callable=run,
    dag=dag,
)