from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from auto_train import main as main_train

params_dict = {
    'meta_path': '/opt/airflow/model_meta/', 
    'data_path': '/opt/airflow/historical_data/', 
    'test_size': 0.001,
    'test_last_fold': False,
    'apply_night_peak': False,
    'start_date': '2023-08-01',
    'end_date': '2200-12-31'
}
train_model_main_path = '/opt/airflow/models_tobe_evaluated/'

def run(params_dict=params_dict, train_model_main_path=train_model_main_path):
    main_train(params_dict, train_model_main_path)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': startdate,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'train_dag',
    default_args=default_args,
    description='Run auto_train.main_train() everyweek',
    schedule_interval='30 0 * * 0',
)

run_p = PythonOperator(
    task_id='run_train',
    python_callable=run,
    dag=dag,
)