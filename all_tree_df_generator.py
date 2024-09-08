import datetime
import pandas as pd
from copy import deepcopy

from data_integration.integrating_power_data import get_full_oneday_power_df

sql_db_fn = './realtime/realtime_data/realtime.db'

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

def get_non_negative_df(sql_db_fn, date):
    df = get_full_oneday_power_df(sql_db_fn, date=date, return_sql_df=True)
    df['時間'] = pd.to_datetime(df['時間'])
    df = df[df['時間'] == max(df['時間'])].reset_index(drop=True)

    for i in df.index:
        if df.loc[i]['機組'] == '其它台電自有':
            df.loc[i, '機組'] += df.loc[i, '分類']
        
    df = df[df['分類'] != '儲能負載(Energy Storage Load)']

    df['電廠'] = [get_plant_name(df.loc[i, '機組'], df.loc[i, '分類']) for i in df.index]

    for i in df.index:
        if df.loc[i, '電廠'] == '大林電廠':
            df.loc[i, '電廠'] += df.loc[i, '分類']

    non_neg_df = deepcopy(df)
    non_neg_df = non_neg_df[non_neg_df['發電量']>=0].drop('時間', axis=1)
    return non_neg_df

def build_hierarchical_dataframe(df, levels, value_column, total_name='total', color_columns=None):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_list = []
    for i, level in enumerate(levels):
        columns = ['id', 'parent', 'value']
        if not color_columns is None:
            columns.append('color')
        df_tree = pd.DataFrame(columns=columns)
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
        else:
            df_tree['parent'] = total_name
        df_tree['value'] = dfg[value_column]
        if not color_columns is None:
            df_tree['color'] = dfg[color_columns[0]] / dfg[color_columns[1]]
        df_list.append(df_tree)
    total_dict = {'id':[total_name],
                  'parent':[''],
                  'value':[df[value_column].sum()]}
    if not color_columns is None:
        total_dict['color'] = [df[color_columns[0]].sum() / df[color_columns[1]].sum()]
    total = pd.DataFrame(total_dict, index=[1000])
    df_list.append(total)
    df_all_trees = pd.concat(df_list, ignore_index=True)
    return df_all_trees

def get_all_tree_df(sql_db_fn=sql_db_fn):
    now = datetime.datetime.now()
    today = now.date()
    if now.hour == 0 and now.minute <= 40:
        today -= datetime.timedelta(days=1)
    
    non_neg_df = get_non_negative_df(sql_db_fn, today)
    tree_df = build_hierarchical_dataframe(df=non_neg_df, levels=['機組', '電廠', '分類'], value_column='發電量', total_name='總發電')
    return tree_df
