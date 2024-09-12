from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from auto_predict import predict

data_path = '/opt/airflow/historical_data/'
model_path = '/opt/airflow/model/'

def run():
    now = datetime.datetime.now()
    if now.time() > datetime.time(19, 10, 0, 0):
        _ = predict(data_path, model_path)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': startdate,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'pred_dag',
    default_args=default_args,
    description='Run auto_predict.predict() everyday',
    schedule_interval='25 19 * * *',
    catchup=False,
)

run_p = PythonOperator(
    task_id='run_pred',
    python_callable=run,
    dag=dag,
)