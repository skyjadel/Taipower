from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from Evaluate_Given_Models import evaluate_models

data_path = '/opt/airflow/historical_data/'
model_dir = '/opt/airflow/models_tobe_evaluated/'
current_model_path = '/opt/airflow/model/'

def run(data_path=data_path, model_dir=model_dir, current_model_path=current_model_path):
    evaluate_models(data_path, model_dir=model_dir, current_model_path=current_model_path)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': startdate,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'eval_models_dag',
    default_args=default_args,
    description='Run Evaluate_Given_Models.evalute_models() everyday',
    schedule_interval='0 2 * * 6',
    catchup=False,
)

run_p = PythonOperator(
    task_id='run_eval_models',
    python_callable=run,
    dag=dag,
)