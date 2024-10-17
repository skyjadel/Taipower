import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
import datetime
import time
import os

from chatbot import respond_generator as chatbot_response
from chatbot import global_agent

#----------Initialization-----------

realtime_data_path = './realtime/realtime_data/'
historical_data_path = './historical/data/prediction/'
historical_power_structure_path = './historical/data/power/power_structure/'
realtime_power_structure_path = f'{historical_power_structure_path}today/'
y_feature_list = ['尖峰負載', '太陽能', '風力']
moving_average_days = 14

now = datetime.datetime.now() 
now += (datetime.timedelta(hours=8) - now.astimezone().tzinfo.utcoffset(None))

first_hour_show_today_peak = 12
first_min_show_lastday_eval = 35
first_min_show_today_pwd_treemap = 25

#   Define power units
units = {
    '風力': 'MW',
    '太陽能': 'MW',
    '尖峰負載': 'GW'
}

#   The original values in the csv files are in the unit of ten-million-watts.
#   We define the converting factors here. 
unit_convert_factor = {
    'KW': 10000,
    'MW': 10,
    'GW': 0.01
}

unit_factor = {}
for k, u in units.items():
    unit_factor[k] = unit_convert_factor[u]

# Read predictions and observations
eval_df = pd.read_csv(f'{historical_data_path}evaluation.csv')
power_pred_df = pd.read_csv(f'{historical_data_path}power.csv')

# Define metric layout
if list(power_pred_df['日期']).index(eval_df['日期'].iloc[-2]) == len(power_pred_df) - 2:
    second_row_first_col = 1
elif list(power_pred_df['日期']).index(eval_df['日期'].iloc[-2]) == len(power_pred_df) - 3:
    second_row_first_col = 2

if now.hour >= first_hour_show_today_peak or ((now.hour == 0 and now.minute <= first_min_show_lastday_eval) and second_row_first_col==2):
    peak_today_df = pd.read_csv(f'{realtime_data_path}peak.csv')
    peak_today = peak_today_df.to_dict(orient='list')

#--------Functions----------

def value_string(val, unit):
    '''
    按照數字的大小決定給多少小數位
    '''
    if val >= 1000:
        return f'{val:.0f} {unit}'
    if val >= 100:
        return f'{val:.1f} {unit}'
    if val >= 10:
        return f'{val:.2f} {unit}'
    return f'{val:.3f} {unit}'


def one_tab(y_feature, second_row_first_col, moving_average_days=moving_average_days):
    '''
    建構風力、尖峰負載、太陽能三個 tab 的完整內容
    '''
    this_unit = units[y_feature]
    this_unit_factor = unit_factor[y_feature]

    true_value_list = [float(eval_df[y_feature].iloc[i]) * this_unit_factor for i in range(len(eval_df) - 1)]
    err = [np.abs(float(eval_df[f'{y_feature}_預測'].iloc[i]) - float(eval_df[y_feature].iloc[i])) * this_unit_factor for i in range(len(eval_df) - 1)]
    avg_err = [np.mean(err[max(0,i-moving_average_days+1):i+1]) for i in range(len(err))]

    left, right = st.columns(2)

    # 左半邊數字部分
    left.header(y_feature)
    left.markdown('#### 最新預測')

    # 第一行：預測值
    first_row = list(left.columns(3))
    for j in range(3):
        first_row[j].metric(
            label=f"{power_pred_df['日期'].iloc[-j-1]} 預測",
            value=value_string(power_pred_df[y_feature].iloc[-j-1] * this_unit_factor, this_unit), 
        )

    left.markdown('#### 近日預測表現')

    # 第二行：實際觀測值
    second_row = list(left.columns(3))
    # 在 12 點後顯示今天的推定數值
    if now.hour >= first_hour_show_today_peak:
        today_str = datetime.datetime.strftime(now, '%Y-%m-%d')
        second_row[second_row_first_col-1].metric(
            label=f"今日 ({today_str}) 推定數據",
            value=value_string(peak_today[y_feature][0] * this_unit_factor, this_unit)
        )
    elif (now.hour == 0 and now.minute <= first_min_show_lastday_eval) and second_row_first_col==2:
        yesterday_str = datetime.datetime.strftime(now - datetime.timedelta(days=1), '%Y-%m-%d')
        second_row[second_row_first_col-1].metric(
            label=f"昨日 ({yesterday_str}) 推定數據",
            value=value_string(peak_today[y_feature][0] * this_unit_factor, this_unit)
        )
    # 過去的實際值
    for i, j in enumerate(range(second_row_first_col, 3)):
        second_row[j].metric(
            label=f"{eval_df['日期'].iloc[-i-2]} 真實數據",
            value=value_string(float(eval_df[y_feature].iloc[-i-2]) * this_unit_factor, this_unit),
        )

    # 第三行：單日誤差
    third_row = list(left.columns(3))
    # 在 12 點後顯示今天的推定誤差
    if now.hour >= first_hour_show_today_peak:
        this_err = np.abs(peak_today[y_feature][0] - power_pred_df[y_feature].iloc[-second_row_first_col]) * this_unit_factor
        last_err = err[-1]
        third_row[second_row_first_col-1].metric(
            label=f"今日 ({today_str}) 推定誤差",
            value=value_string(this_err, this_unit),
            delta=value_string(this_err - last_err, this_unit),
            delta_color='inverse',
        )
    elif (now.hour == 0 and now.minute <= first_min_show_lastday_eval) and second_row_first_col==2:
        this_err = np.abs(peak_today[y_feature][0] - power_pred_df[y_feature].iloc[-second_row_first_col]) * this_unit_factor
        last_err = err[-1]
        third_row[second_row_first_col-1].metric(
            label=f"昨日 ({yesterday_str}) 推定誤差",
            value=value_string(this_err, this_unit),
            delta=value_string(this_err - last_err, this_unit),
            delta_color='inverse',
        )
    # 過去的誤差
    for i, j in enumerate(range(second_row_first_col, 3)):
        this_err = err[-i-1]
        last_err = err[-i-2]
        third_row[j].metric(
            label=f"{eval_df['日期'].iloc[-i-2]} 預測誤差",
            value=value_string(this_err, this_unit),
            delta=value_string(this_err - last_err, this_unit),
            delta_color='inverse',
        )

    # 第四行：移動平均誤差
    forth_row = list(left.columns(3))
    # 12 點以後顯示當日到目前為止的尖峰時間
    if now.hour >= first_hour_show_today_peak:
        forth_row[second_row_first_col-1].metric(
            label=f"今日尖峰時間",
            value=peak_today['尖峰時間'][0].split(' ')[1]
        )
    elif (now.hour == 0 and now.minute <= first_min_show_lastday_eval) and second_row_first_col==2:
        forth_row[second_row_first_col-1].metric(
            label=f"昨日尖峰時間",
            value=peak_today['尖峰時間'][0].split(' ')[1]
        )
    # 過去的移動平均誤差
    for i, j in enumerate(range(second_row_first_col, 3)):
        this_val = avg_err[-i-1]
        last_val = avg_err[-i-2]
        forth_row[j].metric(
            label=f"{eval_df['日期'].iloc[-i-2]}, {moving_average_days}日內平均誤差",
            value=value_string(this_val, this_unit),
            delta=value_string(this_val - last_val, this_unit),
            delta_color='inverse',
        )

    # 第五行：歷史標準差
    fifth_row = list(left.columns(3))
    fifth_row[0].metric(
        label='真實數據歷史標準差',
        value=value_string(float(eval_df[f'{y_feature}7日內平均誤差'].iloc[-1]) * this_unit_factor, this_unit),
    )

    fifth_row[1].metric(
        label=f'{moving_average_days}日內真實數據標準差',
        value=value_string(np.std(true_value_list[-moving_average_days::]), this_unit)
    )
    # 備註
    left.text('台灣時間每日 00:30 更新前一日真實發電量資料與預測表現, 每日 19:30 更新次日發電量預測')

    # 右半邊圖表部分
    right.markdown('#### 歷史預測表現')
 
    df = deepcopy(eval_df[0:-1])
    if len(df) > 30: # 圖最多顯示 30 天的數據
        df = df[-30::]
        avg_err = avg_err[-30::]
    df['日期'] = pd.to_datetime(df['日期'])
    df[f'{y_feature}'] = [float(v) * this_unit_factor for v in df[f'{y_feature}']]
    df[f'{y_feature}_預測'] = [float(v) * this_unit_factor for v in df[f'{y_feature}_預測']]
    df[f'{y_feature}平均誤差'] = avg_err[-30::]
    df[f'{y_feature}誤差'] = np.abs(df[f'{y_feature}_預測'] - df[f'{y_feature}'])

    line_fig = px.line(df, x='日期', y=f'{y_feature}平均誤差', color_discrete_sequence=['gray'],
                       labels={f'{y_feature}平均誤差': f'{moving_average_days}日平均誤差'})
    
    # 誤差移動平均圖
    fig = go.Figure()
    for trace in line_fig.data:
        fig.add_trace(trace)

    fig.update_layout(
        title=f'{moving_average_days}日平均誤差 ({this_unit})',
        xaxis_title='日期',
        yaxis_title=f'{y_feature}誤差',
    )

    fig.update_traces(selector=dict(type='scatter'), mode='lines+markers+text', textposition="top center")
    right.plotly_chart(fig)

    # 預測值與真實值的比對圖
    new_dict = {
        '日期': [],
        '類型': [],
        '發電量': []
    }

    for i in range(len(df)):
        new_dict['日期'].append(df['日期'].iloc[i])
        new_dict['類型'].append('預測')
        new_dict['發電量'].append(df[f'{y_feature}_預測'].iloc[i])

        new_dict['日期'].append(df['日期'].iloc[i])
        new_dict['類型'].append('真實數據')
        new_dict['發電量'].append(df[f'{y_feature}'].iloc[i])

    new_df = pd.DataFrame(new_dict)

    fig = px.line(new_df, x='日期', y='發電量', color='類型', title=f'{y_feature}的預測與真實數據 ({this_unit})')
    fig.update_traces(mode='lines+markers+text', textposition="top center")
    right.plotly_chart(fig)


# 製作聊天機器人
def AI_assistant():

    avatar_dict = {
        'user': '🕵️‍♂️',
        'assistant': '🤖'
    }

    # 因為預期有很多問題會需要讓 LLM 知道今天是幾月幾號，所以把這個訊息塞到 prompt 裡面
    def prompt_generator(input_message):
        return f'背景知識：今天是{now.date()}, 對話內容：{input_message}'
    
    # Streamed response emulator
    def response_generator(input_message):
        prompt = prompt_generator(input_message)
        response = chatbot_response(prompt, global_agent)
        for word in response:
            yield word
            time.sleep(0.05)

    st.header('聊天機器人')
    st.markdown('#### 問題範例')
    st.text('上個月太陽能發電的總占比是百分之多少，核能發電的總占比又是百分之多少？')
    st.text('今天的風力發電預計佔尖峰發電量多少比例？')
    st.text('過去七天當中，太陽能發電預測的誤差平均多少？')
    st.text('過去七天當中，哪一天的風力發電預測誤差最小，那天的預測值、實際值與誤差分別是多少？')
    st.text('整個八月中旬，台電的尖峰負載平均有多少？')
    st.text('昨天發電量前五名的電廠是哪些？')
    st.text('過去五天之內，台中電廠平均每天的發電量是多少？')

    # 初始化聊天歷史紀錄器
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 把過去聊天內容印出來
    for _, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar=avatar_dict[message["role"]]):
            st.markdown(message["content"])

    # 目前對話環節
    if prompt := st.chat_input("可以問我關於台電太陽能與風力發電的預測與實際資料，以及尖峰負載的相關問題。"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        role = 'user'
        with st.chat_message(role, avatar=avatar_dict[role]):
            st.markdown(prompt)

        role = 'assistant'
        with st.chat_message(role, avatar=avatar_dict[role]):
            response = st.write_stream(response_generator(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.text('Powered by OpenAI GPT-4')


# 製作發電結構圖
def tree_map():
    max_depth_dict = {
        '發電方式分類': 2,
        '電廠': 3,
        '機組': 4
    }
    left, right = st.columns(2)


    # 左半邊
    left.markdown('# 即時供電結構')

    # 準備時間選項與對應資料檔案名
    time_filename_dict = {datetime.datetime.strptime(s.split('.')[0], '%Y-%m-%d_%H-%M'): s for s in os.listdir(f'{realtime_power_structure_path}')}
    time_str_dict = {datetime.datetime.strftime(t, '%H:%M'):t for t in time_filename_dict.keys()}
    time_option_list = list(time_str_dict.keys())

    # 如果原始時間選項裡面有 23:59 或 00:00，把它們都換成 24:00
    for replace_time in ['23:59', '00:00']:
        if replace_time in time_option_list:
            time_option_list.remove(replace_time)
            if not '24:00' in time_option_list:
                time_option_list.append('24:00')
            time_str_dict['24:00'] = time_str_dict[replace_time]

    time_option_list.sort()

    # 時間與層數選項
    left_a, left_b = left.columns(2)

    time_str = left_a.selectbox(
        label='請選擇時間',
        options=time_option_list,
        index=len(time_option_list)-1,
        )
    
    max_depth_label_l = left_b.selectbox(
        label='請選擇最小單位',
        options=['發電方式分類', '電廠', '機組'],
        index=2,
        key='deepest_l',
        )
    
    # 按照上面的時間選項讀取對應檔案
    df_filename = f'{realtime_power_structure_path}{time_filename_dict[time_str_dict[time_str]]}'
    df_now = pd.read_csv(df_filename)

    total = df_now[df_now['id']=='即時總發電功率'].iloc[0]['value']
    df_now['百分比'] = [f'{df_now["value"].iloc[i]/total*100:.2f}%' for i in range(len(df_now))]

    # 畫出結構圖
    fig1 = go.Figure()
    fig1.add_trace(go.Treemap(
         labels=df_now['id'],
         parents=df_now['parent'],
         values=df_now['value'],
         customdata=df_now['百分比'],
         branchvalues='total',
         marker=dict(colors=df_now['color']),
         hovertemplate='<b>%{label} </b> <br> 即時供電功率: %{value:.1f} MW<br> 百分比: %{customdata}',
         texttemplate="%{label}<br>%{value:.1f} MW<br>%{customdata}",
         textposition='middle center',
         textfont_size=16,
         maxdepth=max_depth_dict[max_depth_label_l],
         name=''
         ))
    fig1.update_layout(
        margin = dict(t=50, l=25, r=25, b=25),
        height = 900
        )
    left.plotly_chart(fig1)


    # 右半邊
    right.markdown('# 全日供電結構')

    # 日期與深度選項
    right_a, right_b = right.columns(2) 
    if now.hour == 0 and now.minute < first_min_show_today_pwd_treemap:
        yesterday = now - datetime.timedelta(days=1)
        input_date = right_a.date_input(label='請輸入日期', value=yesterday, min_value=datetime.date(2024,8,1), max_value=yesterday)
    else:
        today = now
        input_date = right_a.date_input(label='請輸入日期', value=today, min_value=datetime.date(2024,8,1), max_value=today)
    date_str = datetime.datetime.strftime(input_date, '%Y-%m-%d')

    max_depth_label_r = right_b.selectbox(
        label='請選擇最小單位',
        options=['發電方式分類', '電廠', '機組'],
        index=2,
        key='deepest_r',
        )

    # 按照上面日期讀取資料
    df_all = pd.read_csv(f'{historical_power_structure_path}{date_str}.csv')
    total = df_all[df_all['id']=='今日總發電量'].iloc[0]['value']
    df_all['百分比'] = [f'{df_all["value"].iloc[i]/total*100:.2f}%' for i in range(len(df_all))]

    # 繪製結構圖
    fig2 = go.Figure()
    fig2.add_trace(go.Treemap(
         labels=df_all['id'],
         parents=df_all['parent'],
         values=df_all['value'],
         customdata=df_all['百分比'],
         branchvalues='total',
         marker=dict(colors=df_all['color']),
         hovertemplate='<b>%{label} </b> <br> 今日總供電量: %{value:.2f} GWh<br>百分比: %{customdata}',
         texttemplate="%{label}<br>%{value:.2f} GWh<br>%{customdata}",
         textposition='middle center',
         textfont_size=16,
         maxdepth=max_depth_dict[max_depth_label_r],
         name=''
         ))
    fig2.update_layout(
        margin = dict(t=50, l=25, r=25, b=25),
        height = 900
        )
    right.plotly_chart(fig2)

#------------Main------------

st.set_page_config(
    page_title='台電綠能尖峰發電預測',  
    page_icon=':high_brightness:',
    layout='wide',
)

st.title('台電綠能尖峰發電預測')
st.text('By Y. W. Liao')
tabs = list(st.tabs(y_feature_list + ['聊天機器人', '發電結構圖']))

for i, tab in enumerate(tabs[0:-2]):
    with tab:
        one_tab(y_feature_list[i], second_row_first_col)

with tabs[-1]:
    tree_map()

with tabs[-2]:
    AI_assistant()


    
