from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from auto_evaluate import evaluate

data_path = '/opt/airflow/historical_data/'
model_path = '/opt/airflow/model/'

def run():
    _ = evaluate(data_path)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': startdate,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'eval_dag',
    default_args=default_args,
    description='Run auto_evaluate.predict() everyday',
    schedule_interval='28 0 * * *',
)

run_eval = PythonOperator(
    task_id='run_eval',
    python_callable=run,
    dag=dag,
)