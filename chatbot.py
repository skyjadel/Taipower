from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from openai import RateLimitError

import pandas as pd
from copy import deepcopy
import numpy as np
import datetime

###################################################
#    定義一個聊天機器人給 dashboard.py 呼叫       #
##################################################

historical_data_path = './historical/data/'
realtime_data_path = './realtime/realtime_data/'
api_key_path = 'D:/資料科學與人工智慧實驗室/授權/openai/key.txt'
model = 'gpt-4'


# 按照尖峰負載、風力與太陽能的預測值與實際值建構參考資料表
def build_rag_df(historical_data_path=historical_data_path):
    eval_df = pd.read_csv(historical_data_path + 'prediction/evaluation.csv')[0:-1]
    eval_df['日期'] = pd.to_datetime(eval_df['日期'])
    pred_df = pd.read_csv(historical_data_path + 'prediction/power.csv')
    pred_df['日期'] = pd.to_datetime(pred_df['日期'])

    for col in eval_df.columns:
        if col != '日期':
            eval_df[col] = [float(eval_df[col].iloc[i]) for i in range(len(eval_df))]

    rag_df = deepcopy(eval_df)

    this_dict = {k:[] for k in rag_df.columns}
    for i in pred_df.index:
        if not pred_df['日期'].loc[i] in list(rag_df['日期']):
            for key in this_dict.keys():
                if key == '日期':
                    this_dict[key].append(pred_df['日期'].loc[i])
                elif '_預測' in key:
                    this_dict[key].append(pred_df[key.split('_')[0]].loc[i])
                else:
                    this_dict[key].append(np.nan)
    rag_df = pd.concat([rag_df, pd.DataFrame(this_dict)], axis=0, ignore_index=True).reset_index(drop=True)

    col_map = {}
    for col in rag_df.columns:
        if not col == '日期':
            if '尖峰負載' in col:
                col_map[col] = col.replace('尖峰負載', '尖峰負載(aka總發電量)') + '(MW)'
            else:
                col_map[col] = col + '(MW)'
            rag_df[col] *= 10

    rag_df.rename(col_map, axis=1, inplace=True)
    return rag_df


# 依照發電結構資料建立參考資料表 (尚未上線)
def build_generator_rag_df(realtime_data_path=realtime_data_path):
    whole_day_df = pd.read_csv(f'{realtime_data_path}whole_day_df.csv')
    realtime_df = pd.read_csv(f'{realtime_data_path}realtime_df.csv')
    rag_df = pd.merge(whole_day_df, realtime_df, on=['分類', '機組', '電廠'])
    rag_df.rename({'總發電量(GWhr)':'今日總發電量(GWhr)'}, axis=1, inplace=True)
    rag_df['日期'] = datetime.datetime.now().date().strftime('%Y-%m-%d')
    return rag_df


# 建立 agent
def create_agent(api_key_path=api_key_path, df=build_rag_df()):
    with open(api_key_path, 'r') as f:
        api_key = f.read()
    
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(
            temperature=0.3,
            model=model,
            api_key=api_key
        ),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    return agent


# 對話機器人，被 dashboard 直接呼叫的部分
def respond_generator(input_message, agent=create_agent()):
    print('='*30)
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    try:
        respond = agent.invoke(input_message)
        return respond['output']
    except RateLimitError:
        return "已超過請求限制，請稍後再試。"
    except Exception as e:
        return f'發生問題：{e}'