import os
import datetime
from glob import glob

from utils.tree_dataframe import get_whole_day_tree_df, get_alltime_today_tree_df

historical_power_structure_path = './historical/data/power/power_structure/'
sql_db_fn = './realtime/realtime_data/realtime.db'

def save_tree_dfs(sql_db_fn=sql_db_fn, historical_power_structure_path=historical_power_structure_path):
    realtime_power_structure_path = f'{historical_power_structure_path}today/'

    os.makedirs(historical_power_structure_path, exist_ok=True)
    os.makedirs(realtime_power_structure_path, exist_ok=True)

    start_date = datetime.date(2024, 8, 1)
    end_date = datetime.date(2200, 12, 31)

    now = datetime.datetime.now()
    today = now.date()
    if now.hour == 0 and now.minute <= 25:
        today -=  datetime.timedelta(days=1)

    end_date = min(end_date, today)
    date_range = int((end_date - start_date) / datetime.timedelta(days=1)) + 1
    date_list = [start_date + datetime.timedelta(days=i) for i in range(date_range)]

    for date in date_list:
        this_filename = f'{historical_power_structure_path}{datetime.datetime.strftime(date, "%Y-%m-%d")}.csv'
        if (not os.path.exists(this_filename)) or date == today:
            print(this_filename)
            df = get_whole_day_tree_df(sql_db_fn=sql_db_fn, date=date)
            df.to_csv(this_filename, encoding='utf-8-sig', index=False)
    
    all_dict = get_alltime_today_tree_df(sql_db_fn)
    legit_filename_list = []
    today_str = datetime.datetime.strftime(today, "%Y-%m-%d")
    for time_str, this_tree_df in all_dict.items():
        this_filename = f'{realtime_power_structure_path}{today_str}_{time_str}.csv'
        this_filename = this_filename.replace(':', '-')
        if not os.path.exists(this_filename):
            this_tree_df.to_csv(this_filename, encoding='utf-8-sig', index=False)
        legit_filename_list.append(this_filename)
    
    for old_filename in glob(f'{realtime_power_structure_path}*'):
        old_filename = old_filename.replace('\\', '/')
        if not old_filename in legit_filename_list:
            os.remove(old_filename)


