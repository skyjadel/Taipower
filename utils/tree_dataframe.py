import datetime
import numpy as np
from copy import deepcopy
import pandas as pd

from data_integration.integrating_power_data import get_full_oneday_power_df

sql_db_fn = '../realtime/realtime_data/realtime.db'
realtime_data_path = '../realtime/realtime_data/'


color_map = {
    '核能(Nuclear)': [[217, 220, 254], [209, 212, 254], [198, 202, 253]],
    '汽電共生(Co-Gen)': [[254, 221, 140], [254, 213, 115], [254, 203, 82]],
    '民營電廠-燃氣(IPP-LNG)': [[255, 186, 255], [255, 171, 255], [255, 151, 255]],
    '燃煤(Coal)': [[244, 142, 125], [242, 117, 96], [239, 85, 59]],
    '太陽能(Solar)': [[85, 221, 185], [48, 214, 170], [0, 204, 150]],
    '燃油(Oil)': [[255, 153, 183], [255, 131, 167], [255, 102, 146]],
    '風力(Wind)': [[182, 232, 128], [196, 236, 152], [182, 232, 128]],
    '輕油(Diesel)': [[132, 132, 132], [156, 156, 156], [132, 132, 132]],
    '水力(Hydro)': [[151, 158, 252], [128, 137, 251], [99, 110, 250]],
    '燃氣(LNG)': [[199, 151, 252], [187, 128, 251], [171, 99, 250]],
    '儲能(Energy Storage System)': [[102, 153, 255], [70, 133, 255], [102, 153, 255]],
    '民營電廠-燃煤(IPP-Coal)': [[255, 193, 145], [255, 179, 121], [255, 161, 90]],
    '其它再生能源(Other Renewable Energy)': [[102, 226, 247], [68, 219, 245], [25, 211, 243]],
    'N/A': [[232, 232, 232], [232, 232, 232], [232, 232, 232]]
}


def color_assign(first_level, depth, color_map=color_map):
    rgb = color_map[first_level][min(2, depth-1)]
    return 'rgb({}, {}, {})'.format(*rgb)


def get_plant_name(generator_name, power_type):
    generator_name = generator_name.replace('amp;', '')
    generator_name = generator_name.replace('amp', '')
    if 'Gas' in generator_name:
        return generator_name.split('Gas')[0] + 'Gas'
    if power_type == '風力(Wind)':
        return generator_name + '風場'
    if power_type in ['其它再生能源(Other Renewable Energy)', '太陽能(Solar)', '汽電共生(Co-Gen)'] or generator_name=='電池':
        return generator_name + '.'
    return generator_name.split('#')[0] + '電廠'


def build_hierarchical_dataframe(df, levels, value_column, total_name='total', color_map=None):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_list = []
    for i, level in enumerate(levels):
        columns = ['id', 'parent', 'value', 'depth', '1st_level']
        if not color_map is None:
            columns.append('color')
        df_tree = pd.DataFrame(columns=columns)
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
            df_tree['1st_level'] = dfg[levels[-1]].copy()
        else:
            df_tree['parent'] = total_name
            df_tree['1st_level'] = df_tree['id']
        df_tree['value'] = dfg[value_column]
        df_tree['depth'] = len(levels) - i
        if not color_map is None:
            df_tree['color'] = [color_assign(df_tree.loc[i]['1st_level'], df_tree.loc[i]['depth']) for i in df_tree.index]
        df_list.append(df_tree)
    total_dict = {'id':[total_name],
                  'parent':[''],
                  'value':[df[value_column].sum()],
                  'depth':[0],
                  '1st_level':['N/A']}
    if not color_map is None:
        total_dict['color'] = [color_assign(total_dict['1st_level'][0], 0)]
    total = pd.DataFrame(total_dict, index=[1000])

    df_list.append(total)
    df_all_trees = pd.concat(df_list, ignore_index=True)
    return df_all_trees


def sql_df_preprocess(sql_db_fn, date=None):
    day_completed = False
    now = datetime.datetime.now()
    if date is None:
        date = now.date() 
        if now.hour == 0 and now.minute <= 25:
            date -= datetime.timedelta(days=1)
            day_completed = True
    else:
        if date < now.date():
            day_completed = True
    
    df = get_full_oneday_power_df(sql_db_fn, date, return_sql_df=True)
    df = df.loc[~df['發電量'].isna()].reset_index(drop=True)
    df['時間'] = pd.to_datetime(df['時間'])

    for i in df.index:
        if df.loc[i]['機組'] == '其它台電自有':
            df.loc[i, '機組'] += df.loc[i, '分類']
        
    df = df[df['分類'] != '儲能負載(Energy Storage Load)']

    time_list = list(set(df['時間']))
    time_list.sort()

    delta_time_list = [time_list[0] - datetime.datetime(time_list[0].year, time_list[0].month, time_list[0].day)]\
    + [time_list[i] - time_list[i-1] for i in range(1, len(time_list))]
    delta_hour_list = [dt/datetime.timedelta(hours=1) for dt in delta_time_list]
    if day_completed:
        delta_hour_list[-1] += 24 - sum(delta_hour_list)

    df['唯一機組名'] = [f"{df.loc[i]['分類']}+{df.loc[i]['機組']}" for i in df.index]
    df['權重'] = [delta_hour_list[time_list.index(df.loc[i]['時間'])] for i in df.index]
    df['Weighted_發電量'] = df['權重'] * df['發電量']
    return df


def get_whole_day_df(sql_db_fn, date=None):
    df = sql_df_preprocess(sql_db_fn, date)

    df = df.groupby('唯一機組名')['Weighted_發電量'].sum().reset_index()
    df['分類'] = [s.split('+')[0] for s in df['唯一機組名']]
    df['機組'] = [s.split('+')[1] for s in df['唯一機組名']]
    df['電廠'] = [get_plant_name(df.loc[i, '機組'], df.loc[i, '分類']) for i in df.index]

    for i in df.index:
        if df.loc[i, '電廠'] == '大林電廠':
            df.loc[i, '電廠'] += df.loc[i, '分類']
    
    df = df[~(df['分類']=='儲能負載(Energy Storage Load)')]
    df = df.rename({'Weighted_發電量': '總發電量(GWhr)'}, axis=1).drop('唯一機組名', axis=1)
    return df


def get_whole_day_tree_df(df=None, sql_db_fn=None, date=None):
    if df is None:
        df = get_whole_day_df(sql_db_fn, date)

    tree_df = build_hierarchical_dataframe(df,
                                           levels=['機組', '電廠', '分類'],
                                           value_column='總發電量(GWhr)',
                                           total_name='今日總發電量',
                                           color_map=color_map)
    tree_df['value'] /= 1000
    return tree_df


def get_realtime_df(sql_db_fn, date=None):
    df = sql_df_preprocess(sql_db_fn, date)

    df = deepcopy(df[df['時間']==max(df['時間'])])
    df['電廠'] = [get_plant_name(df.loc[i, '機組'], df.loc[i, '分類']) for i in df.index]
    for i in df.index:
        if df.loc[i, '電廠'] == '大林電廠':
            df.loc[i, '電廠'] += df.loc[i, '分類']
    
    df = df[~(df['分類']=='儲能負載(Energy Storage Load)')]
    df = df.rename({'發電量': '即時發電功率(MW)'}, axis=1).drop(['唯一機組名', '時間', 'Weighted_發電量', '容量', '權重'], axis=1).reset_index(drop=True)
    return df


def get_realtime_tree_df(df=None, sql_db_fn=None, date=None):
    if df is None:
        df = get_realtime_df(sql_db_fn, date)

    tree_df = build_hierarchical_dataframe(df=df,
                                           levels=['機組', '電廠', '分類'],
                                           value_column='即時發電功率(MW)',
                                           total_name='即時總發電功率',
                                           color_map=color_map)
    return tree_df


def save_tree_dfs(sql_db_fn, realtime_data_path, date=None):
    whole_day_df = get_whole_day_df(sql_db_fn, date)
    realtime_df = get_realtime_df(sql_db_fn, date)
    whole_day_tree_df = get_whole_day_tree_df(whole_day_df)
    realtime_tree_df = get_realtime_tree_df(realtime_df)

    whole_day_df.to_csv(f'{realtime_data_path}whole_day_df.csv', index=False, encoding='utf-8-sig')
    realtime_df.to_csv(f'{realtime_data_path}realtime_df.csv', index=False, encoding='utf-8-sig')
    whole_day_tree_df.to_csv(f'{realtime_data_path}whole_day_tree_df.csv', index=False, encoding='utf-8-sig')
    realtime_tree_df.to_csv(f'{realtime_data_path}realtime_tree_df.csv', index=False, encoding='utf-8-sig')

