from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from openai import RateLimitError

import datetime
import os

api_key_path = 'D:/資料科學與人工智慧實驗室/授權/openai/key.txt'
if not os.path.exists(api_key_path):
    api_key_path = './openai/key.txt'
model = 'gpt-4o-mini'
#model = 'gpt-4'

sql_db_fn = './historical/data/power/power_structure/RAG_sql.db'

system_message = (
    '指示: 請將日期改成 YYYY-MM-DD 格式, 時間改成 HH:MM:SS 格式 '
    #'資料庫裡面有四個表: peak_load_predition, peak_load_truth, real_time_power_generation, total_daily_power_generation '
    #'total_daily_power_generation 裡面的欄位是 date, unit, plant, type, daily_generation_GWh, 分別代表日期、發電機組、電廠、發電方式、全天總發電量單位GWh '
    #'real_time_power_generation 裡面的欄位是 time, unit, plant, type, realtime_generation_MW, 分別代表時間、發電機組、電廠、發電方式、即時發電功率單位MW '
    #'peak_load_predition 裡面的欄位是 date, wind_energy_MW, solar_energy_MW, total_MW, 分別代表日期、風力發電預測值、太陽能發電預測值、尖峰負載預測值，數值單位都是 MW '
    #'peak_load_truth 裡面的欄位是 date, wind_energy_MW, solar_energy_MW, total_MW, wind_energy_error_MW, solar_energy_error_MW, total_error_MW, '
    '分別代表日期、風力發電實際值、太陽能發電實際值、尖峰負載實際值，風力發電預測誤差、太陽能發電預測誤差、尖峰負載預測誤差，數值單位都是 MW '
    '請將回應數字四捨五入到小數點以下第二位 '
    '問題中有現在或即時，請先找到在 real_time_power_generation 裡面最晚的時間，再去找這個時間的資料'
    '如果在資料庫中找不到完全一樣的文字資料，請找最接近的 '
    '請使用使用者提問的語言回答 '
    '當被問到某種發電方式或某個電廠的發電量，請把符合的值加總起來 '
    '使用者問預測誤差的時候，代表的是實際值與預測值相減後取絕對值 '
    '中文與英數符號之間要有空格 '
    'Gas 在表中的寫法是 LNG '
    '中文的「尖峰」對應到 peak_load_predition, peak_load_truth 兩張表 '
    '如果問到預測，請不要去找 real_time_power_generation, total_daily_power_generation '
    '當被問到電廠發電量或發電方式發電量的排名問題，請將屬於每個電廠或發電方式的資料分別加總，再來排名 '
)


class SQL_Agent():
    def __init__(self, sql_db_fn=sql_db_fn, verbose=False):
        self.Agent = self._create_agent(sql_db_fn, verbose=verbose)
        self.system_message = system_message

    def _create_agent(self, sql_db_fn, model=model, api_key_path=api_key_path, verbose=True):
        engine = create_engine(f'sqlite:///{sql_db_fn}')
        db = SQLDatabase(engine)
        
        with open(api_key_path, 'r') as f:
            api_key = f.read()
        
        llm = ChatOpenAI(temperature=0, model=model, api_key=api_key)
        
        memory = ConversationBufferMemory(return_messages=True)
        agent = create_sql_agent(llm=llm, db=db, verbose=verbose, handle_parsing_errors=True, memory=memory)
        return agent

    def invoke(self, input_message):
        respond = self.Agent.invoke(f'User: "{input_message}", Time Now: {datetime.datetime.now()}, instruction:{system_message}')
        return respond['output']


def respond_generator(input_message, agent=SQL_Agent()):
    print('='*30)
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    try:
        return agent.invoke(input_message)
    except RateLimitError:
        return "已超過請求限制，請稍後再試。"
    except Exception as e:
        return f'發生問題：{e}'