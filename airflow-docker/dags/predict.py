from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime
from datetime import timedelta
from StartDate import startdate
import sys
sys.path.append('/opt/airflow/')
from integrating_realtime_data import main as integration_main
from auto_predict import predict

sql_db_fn = '/opt/airflow/sql_db/realtime.db'
data_path = '/opt/airflow/historical_data/'
model_path = '/opt/airflow/model/'

def run_integration():
    integration_main(sql_db_fn, data_path)

def run_prediction():
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
    description='Run prediction everyday',
    schedule_interval='25 19 * * *',
    catchup=False,
)

run_integration = PythonOperator(
    task_id='run_integration',
    python_callable=run_integration,
    dag=dag
)

run_p = PythonOperator(
    task_id='run_pred',
    python_callable=run_prediction,
    dag=dag,
)

run_integration >> run_p