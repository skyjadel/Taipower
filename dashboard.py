import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
import datetime
import time

from chatbot import respond_generator as chatbot_response

#----------Initialization-----------

realtime_data_path = './realtime/realtime_data/'
historical_data_path = './historical/data/prediction/'
y_feature_list = ['尖峰負載', '風力', '太陽能']
moving_average_days = 14

now = datetime.datetime.now() 
now += (datetime.timedelta(hours=8) - now.astimezone().tzinfo.utcoffset(None))

first_hour_show_today_peak = 12

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

if now.hour >= first_hour_show_today_peak or ((now.hour == 0 and now.minute <= 40) and second_row_first_col==2):
    peak_today_df = pd.read_csv(f'{realtime_data_path}peak.csv')
    peak_today = peak_today_df.to_dict(orient='list')

#--------Functions----------

def value_string(val, unit):
    if val >= 1000:
        return f'{val:.0f} {unit}'
    if val >= 100:
        return f'{val:.1f} {unit}'
    if val >= 10:
        return f'{val:.2f} {unit}'
    return f'{val:.3f} {unit}'

def one_tab(y_feature, second_row_first_col, moving_average_days=moving_average_days):
    this_unit = units[y_feature]
    this_unit_factor = unit_factor[y_feature]

    err = [np.abs(float(eval_df[f'{y_feature}_預測'].iloc[i]) - float(eval_df[y_feature].iloc[i])) * this_unit_factor for i in range(len(eval_df) - 1)]
    avg_err = [np.mean(err[max(0,i-moving_average_days+1):i+1]) for i in range(len(err))]
    #avg_err = [float(eval_df[f'{y_feature}7日內平均誤差'].iloc[i]) * this_unit_factor for i in range(len(eval_df) - 1)]

    left, right = st.columns(2)

    left.header(y_feature)
    left.markdown('#### 最新預測')

    first_row = list(left.columns(4))
    for j in range(4):
        first_row[j].metric(
            label=f"{power_pred_df['日期'].iloc[-j-1]} 預測",
            value=value_string(power_pred_df[y_feature].iloc[-j-1] * this_unit_factor, this_unit), 
        )

    left.markdown('#### 近日預測表現')

    second_row = list(left.columns(4))

    if now.hour >= first_hour_show_today_peak:
        today_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        second_row[second_row_first_col-1].metric(
            label=f"今日 ({today_str}) 推定數據",
            value=value_string(peak_today[y_feature][0] * this_unit_factor, this_unit)
        )
    elif (now.hour == 0 and now.minute <= 40) and second_row_first_col==2:
        yesterday_str = datetime.datetime.strftime(datetime.datetime.now() - datetime.timedelta(days=1), '%Y-%m-%d')
        second_row[second_row_first_col-1].metric(
            label=f"昨日 ({yesterday_str}) 推定數據",
            value=value_string(peak_today[y_feature][0] * this_unit_factor, this_unit)
        )

    for i, j in enumerate(range(second_row_first_col, 4)):
        second_row[j].metric(
            label=f"{eval_df['日期'].iloc[-i-2]} 真實數據",
            value=value_string(float(eval_df[y_feature].iloc[-i-2]) * this_unit_factor, this_unit),
        )

    third_row = list(left.columns(4))

    if now.hour >= first_hour_show_today_peak:
        this_err = np.abs(peak_today[y_feature][0] - power_pred_df[y_feature].iloc[-second_row_first_col]) * this_unit_factor
        last_err = err[-1]
        third_row[second_row_first_col-1].metric(
            label=f"今日 ({today_str}) 推定誤差",
            value=value_string(this_err, this_unit),
            delta=value_string(this_err - last_err, this_unit),
            delta_color='inverse',
        )
    elif (now.hour == 0 and now.minute <= 40) and second_row_first_col==2:
        this_err = np.abs(peak_today[y_feature][0] - power_pred_df[y_feature].iloc[-second_row_first_col]) * this_unit_factor
        last_err = err[-1]
        third_row[second_row_first_col-1].metric(
            label=f"昨日 ({yesterday_str}) 推定誤差",
            value=value_string(this_err, this_unit),
            delta=value_string(this_err - last_err, this_unit),
            delta_color='inverse',
        )

    for i, j in enumerate(range(second_row_first_col, 4)):
        this_err = err[-i-1]
        last_err = err[-i-2]
        third_row[j].metric(
            label=f"{eval_df['日期'].iloc[-i-2]} 預測誤差",
            value=value_string(this_err, this_unit),
            delta=value_string(this_err - last_err, this_unit),
            delta_color='inverse',
        )

    forth_row = list(left.columns(4))

    if now.hour >= first_hour_show_today_peak:
        forth_row[second_row_first_col-1].metric(
            label=f"今日尖峰時間",
            value=peak_today['尖峰時間'][0].split(' ')[1]
        )
    elif (now.hour == 0 and now.minute <= 40) and second_row_first_col==2:
        forth_row[second_row_first_col-1].metric(
            label=f"昨日尖峰時間",
            value=peak_today['尖峰時間'][0].split(' ')[1]
        )
        

    for i, j in enumerate(range(second_row_first_col, 4)):
        this_val = avg_err[-i-1]
        last_val = avg_err[-i-2]
        forth_row[j].metric(
            label=f"{eval_df['日期'].iloc[-i-2]}, {moving_average_days}日內平均誤差",
            value=value_string(this_val, this_unit),
            delta=value_string(this_val - last_val, this_unit),
            delta_color='inverse',
        )

    fifth_row = list(left.columns(4))
    fifth_row[0].metric(
        label='真實數據歷史標準差',
        value=value_string(float(eval_df[f'{y_feature}7日內平均誤差'].iloc[-1]) * this_unit_factor, this_unit),
    )

    right.markdown('#### 歷史預測表現')

    df = deepcopy(eval_df[0:-1])
    if len(df) > 30:
        df = deepcopy(df[-30::])
        avg_err = avg_err[-30::]
    df['日期'] = pd.to_datetime(df['日期'])
    df[f'{y_feature}'] = [float(v) * this_unit_factor for v in df[f'{y_feature}']]
    df[f'{y_feature}_預測'] = [float(v) * this_unit_factor for v in df[f'{y_feature}_預測']]
    #df[f'{y_feature}7日內平均誤差'] = [float(v) * this_unit_factor for v in df[f'{y_feature}7日內平均誤差']]
    df[f'{y_feature}平均誤差'] = avg_err[-30::]
    df[f'{y_feature}誤差'] = np.abs(df[f'{y_feature}_預測'] - df[f'{y_feature}'])

    line_fig = px.line(df, x='日期', y=f'{y_feature}平均誤差', color_discrete_sequence=['gray'],
                       labels={f'{y_feature}平均誤差': f'{moving_average_days}日平均誤差'})

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


    left.text('台灣時間每日 00:30 更新前一日真實發電量資料與預測表現, 每日 19:30 更新次日發電量預測')

def AI_assistant():
    def prompt_generator(input_message):
        return f'背景知識：今天是{datetime.datetime.now().date()}, 對話內容：{input_message}'
    # Streamed response emulator
    def response_generator(input_message):
        prompt = prompt_generator(input_message)
        response = chatbot_response(prompt)
        for word in response:
            yield word
            time.sleep(0.05)


    st.header('聊天機器人')
    st.markdown('#### 問題範例')
    st.text('今天的風力發電預計佔總發電量多少比例？')
    st.text('過去七天當中，太陽能發電預測的誤差平均多少？')
    st.text('過去七天當中，哪一天的風力發電預測誤差最小，那天的預測值與實際值分別多少？')
    st.text('整個八月中旬，台電的尖峰負載平均有多少？')
    #st.text('如果發現機器人給的資訊好像過時了，請重新整理頁面，讓機器人回去資料庫看看最新數據。')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for _, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("可以問我關於台電太陽能與風力發電的預測與實際資料，以及尖峰負載的相關問題。"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})

def tree_map():
    
    left, right = st.columns(2)

    df_now = pd.read_csv(f'{realtime_data_path}realtime_tree_df.csv')
    total = df_now[df_now['id']=='即時總發電功率'].iloc[0]['value']
    df_now['百分比'] = [f'{df_now["value"].iloc[i]/total*100:.2f}%' for i in range(len(df_now))]

    left.markdown('# 即時發電結構')
    fig1 = go.Figure()
    fig1.add_trace(go.Treemap(
         labels=df_now['id'],
         parents=df_now['parent'],
         values=df_now['value'],
         customdata=df_now['百分比'],
         branchvalues='total',
         marker=dict(colors=df_now['color']),
         hovertemplate='<b>%{label} </b> <br> 即時發電功率: %{value:.1f} MW<br> 百分比: %{customdata}',
         texttemplate="%{label}<br>%{value:.1f} MW<br>%{customdata}",
         textposition='middle center',
         textfont_size=16,
         name=''
         ))
    fig1.update_layout(
        margin = dict(t=50, l=25, r=25, b=25),
        height = 900
        )
    left.plotly_chart(fig1)

    df_all = pd.read_csv(f'{realtime_data_path}whole_day_tree_df.csv')
    total = df_all[df_all['id']=='今日總發電量'].iloc[0]['value']
    df_all['百分比'] = [f'{df_all["value"].iloc[i]/total*100:.2f}%' for i in range(len(df_all))]

    right.markdown('# 今日總發電量結構')
    fig2 = go.Figure()
    fig2.add_trace(go.Treemap(
         labels=df_all['id'],
         parents=df_all['parent'],
         values=df_all['value'],
         customdata=df_all['百分比'],
         branchvalues='total',
         marker=dict(colors=df_all['color']),
         hovertemplate='<b>%{label} </b> <br> 今日總發電量: %{value:.2f} GWhr<br>百分比: %{customdata}',
         texttemplate="%{label}<br>%{value:.2f} GWhr<br>%{customdata}",
         textposition='middle center',
         textfont_size=16,
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


    
