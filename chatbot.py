from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from openai import RateLimitError

import threading
import logging
import time
import atexit

import datetime
import os

api_key_path = 'D:/資料科學與人工智慧實驗室/授權/openai/key.txt'
if not os.path.exists(api_key_path):
    api_key_path = './openai/key.txt'
model = 'gpt-4o-mini'
#model = 'gpt-4'

sql_db_fn = './historical/data/power/power_structure/RAG_sql.db'

now = datetime.datetime.now() 
now += (datetime.timedelta(hours=8) - now.astimezone().tzinfo.utcoffset(None))

system_message = (
    '指示: 請將日期改成 YYYY-MM-DD 格式, 時間改成 HH:MM:SS 格式 '
    '資料庫裡面有四個表: peak_load_predition, peak_load_truth, real_time_power_generation, total_daily_power_generation '
    'total_daily_power_generation 裡面的欄位是 date, unit, plant, type, daily_generation_GWh, 分別代表日期、發電機組、電廠、發電方式、全天總發電量單位GWh '
    'real_time_power_generation 裡面的欄位是 time, unit, plant, type, realtime_generation_MW, 分別代表時間、發電機組、電廠、發電方式、即時發電功率單位MW '
    'peak_load_predition 裡面的欄位是 date, wind_energy_MW, solar_energy_MW, total_MW, 分別代表日期、風力發電預測值、太陽能發電預測值、尖峰負載預測值，數值單位都是 MW '
    'peak_load_truth 裡面的欄位是 date, wind_energy_MW, solar_energy_MW, total_MW, wind_energy_error_MW, solar_energy_error_MW, total_error_MW, '
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
    '如果問到預測、預計，請到 peak_load_predition找資料'
    '當被問到電廠發電量或發電方式發電量的排名問題，請將屬於每個電廠或發電方式的資料分別加總，再來排名 '
)

# 設置日誌
logging.basicConfig(
    filename='./chatbot.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class SQL_Agent():
    def __init__(self, sql_db_fn=sql_db_fn, api_key_path=api_key_path,
                 model=model, system_message=system_message,
                 verbose=True, check_interval=60):
        
        self.sql_db_fn = sql_db_fn
        self.verbose = verbose
        self.system_message = system_message
        self.api_key_path = api_key_path
        self.model = model
        self.check_interval = check_interval
        self.last_mod_time = None
        self.Agent = None
        self.lock = threading.Lock()
        self._refresh_agent()

        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_db_changes, daemon=True)
        self.monitor_thread.start()
        atexit.register(self.stop)

    def _create_agent(self):
        try:
            engine = create_engine(f'sqlite:///{self.sql_db_fn}')
            db = SQLDatabase(engine)
            
            with open(self.api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            llm = ChatOpenAI(temperature=0, model=self.model, api_key=api_key)
            
            memory = ConversationBufferMemory(return_messages=True)
            agent = create_sql_agent(llm=llm, db=db, verbose=self.verbose, handle_parsing_errors=True, memory=memory)
            logging.info('Agent 創建成功。')
            return agent
        except Exception as e:
            logging.error(f'Agent 創建失敗: {e}')
            return None
    
    def _refresh_agent(self):
        with self.lock:
            self.Agent = self._create_agent()
            self.last_mod_time = os.path.getmtime(self.sql_db_fn)
            logging.info(f"Agent 已刷新。資料庫最後修改時間：{self.last_mod_time}")

    def _monitor_db_changes(self):
        while not self.stop_event.is_set():
            try:
                current_mod_time = os.path.getmtime(self.sql_db_fn)
                if self.last_mod_time is None or current_mod_time != self.last_mod_time:
                    logging.info('資料庫已更新，重開 Agent。')
                    self._refresh_agent()
            except Exception as e:
                logging.error(f"監控資料庫變更時發生錯誤：{e}")
            time.sleep(self.check_interval)

    def invoke(self, input_message):
        with self.lock:
            if self.Agent is None:
                logging.error("Agent 尚未創建。")
                return "聊天機器人初始化中，請稍後再試。"
            respond = self.Agent.invoke(f'User: "{input_message}", Time Now: {now}, instruction:{system_message}')
            return respond['output']
    
    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()
        logging.info("Agent 停止。")

global_agent = SQL_Agent()

def respond_generator(input_message, agent_instance):
    logging.info('='*30)
    logging.info(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    try:
        return agent_instance.invoke(input_message)
    except RateLimitError:
        return "已超過請求限制，請稍後再試。"
    except Exception as e:
        return f'發生問題：{e}'