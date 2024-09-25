# 從台電網站抓取即時電力資料

import json
import requests
from bs4 import BeautifulSoup as bs
import sqlite3
from datetime import timedelta, datetime

def get_data(sql_db_path):
    # 抓取最新資料
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    headers = {"User-Agent": user_agent}
    req = requests.get(url='https://www.taipower.com.tw/d006/loadGraph/loadGraph/data/genary.json', headers=headers)
    req.encoding = 'utf-8'

    JS = json.loads(req.text)

    time_str = JS['']
    power_data = JS['aaData']

    # 如果資料時間為 0 點整，則將它視為前一天的資料，時間改為前一天的 23:59，以利後續整理
    time_now = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
    if time_now.hour == 0 and time_now.minute == 0:
        time_now -= timedelta(minutes=1)
        time_str = datetime.strftime(time_now, '%Y-%m-%d %H:%M') 

    # 存入資料庫
    conn = sqlite3.connect(sql_db_path)
    cursor = conn.cursor()

    sql_command = (
        'CREATE TABLE IF NOT EXISTS power('
        'time DATETIME,'
        'type VARCHAR(50),'
        'generator VARCHAR(50),'
        'volume FLOAT,'
        'produce FLOAT'
        ')'
    )
    cursor.execute(sql_command)
    conn.commit()

    cursor.execute(f"SELECT time, type, generator FROM power WHERE time = '{time_str}'")
    existing_data = cursor.fetchall()

    for row in power_data:
        power_generator = row[2].split('(')[0]
        if not power_generator == '小計':
            power_type = bs(row[0], 'html.parser').find('b').text
            power_volume = 'NULL' if row[3] == '-' else float(row[3])
            power_produce = 'NULL' if row[4] in ['-', 'N/A'] else float(row[4])
            if not (time_str, power_type, power_generator) in existing_data:
                cursor.execute(f"INSERT INTO power VALUES ('{time_str}', '{power_type}', '{power_generator}', {power_volume}, {power_produce});")

    conn.commit()
    cursor.close()
    conn.close()