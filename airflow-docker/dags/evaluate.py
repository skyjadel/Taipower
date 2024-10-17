from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from integrating_realtime_data import main as integration_main
from auto_evaluate import evaluate

sql_db_fn = '/opt/airflow/sql_db/realtime.db'
data_path = '/opt/airflow/historical_data/'
model_path = '/opt/airflow/model/'

def run_integration():
    integration_main(sql_db_fn, data_path)

def run_eval():
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
    description='Run model evaluation everyday',
    schedule_interval='28 0 * * *',
    catchup=False,
)

run_integration = PythonOperator(
    task_id='run_integration',
    python_callable=run_integration,
    dag=dag
)

run_eval = PythonOperator(
    task_id='run_eval',
    python_callable=run_eval,
    dag=dag,
)

run_integration >> run_eval