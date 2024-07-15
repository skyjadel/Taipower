from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as bs
import datetime
import pandas as pd
import sqlite3

station_id_table = {
    '臺北':'46692',
    '嘉義':'46748',
    '高雄':'46744',
    '東吉島':'46730',
    '臺西':'C0K53',
    '臺中電廠':'C0FA4'
}

def set_chrome_driver():

    service = Service(executable_path=ChromeDriverManager().install())
    
    # 這些建議都加上，不開頁面、禁用GPU加速等等
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  
    options.add_argument("--disable-gpu") 
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    
    driver = webdriver.Chrome(service=service, options=options)
    
    #print("Chrome version:", driver.capabilities['browserVersion'])
    #print("ChromeDriver version:", driver.capabilities['chrome']['chromedriverVersion'])

    return driver

def get_weather_observation(station_name):
    station_id = station_id_table[station_name]
    isfull_station = station_id.isdigit()

    row_num = 48 if isfull_station else 24
    
    driver = set_chrome_driver()
    url = f'https://www.cwa.gov.tw/V8/C/W/OBS_Station.html?ID={station_id}'

    driver.get(url)
    driver.implicitly_wait(20)
    soup = bs(driver.page_source.encode('utf-8'), 'html.parser')

    table = soup.find('table').find('tbody')
    table_rows = table.find_all('tr')

    time_now = datetime.datetime.now()
    year_now = time_now.year
    month_now = time_now.month

    data = {'測站':[station_name] * row_num,
            '時間':[],
            '溫度':[],
            '氣候狀況':[],
            '風向':[],
            '風速':[],
            '陣風':[],
            '相對溼度':[],
            '海平面氣壓':[],
            '累積雨量':[],
            '累積日照':[]
           }

    def nan_int(this_str):
        if this_str == '-':
            return 'Null'
        return int(this_str)
    
    def nan_float(this_str):
        if this_str == '-':
            return 'Null'
        return float(this_str)
    
    def get_date_from_span(td, data_type='float'):
        span = td.find('span')
        if not span is None:
            text = span.text
            if data_type == 'int':
                return nan_int(text)
            elif data_type == 'float':
                return nan_float(text)
            else:
                if text == '-':
                    return 'Null'
                return text
        return 'Null'
    
    row_count = 0
    for row in table_rows:
        time_str = row.find('th').get_text()
        if time_str[-2::] in ['00', '30'] and row_count < row_num:
            this_month = int(time_str[0:2])
            this_year = year_now if month_now >= this_month else year_now - 1
            time_str = str(this_year) + '/'+ time_str + ':00'
            data['時間'].append(time_str)
            data['溫度'].append(get_date_from_span(row.find_all('td')[0], data_type='float'))
            img_tag = row.find_all('td')[1].find('img')
            if img_tag is None:
                data['氣候狀況'].append('Null')
            else:
                data['氣候狀況'].append(img_tag.attrs['title'])
            data['風向'].append(get_date_from_span(row.find_all('td')[2], data_type='str'))
            data['風速'].append(get_date_from_span(row.find_all('td')[3], data_type='float'))
            data['陣風'].append(get_date_from_span(row.find_all('td')[4], data_type='float'))
            data['相對溼度'].append(nan_int(row.find_all('td')[6].text))
            data['海平面氣壓'].append(nan_float(row.find_all('td')[7].text))
            data['累積雨量'].append(nan_float(row.find_all('td')[8].text))
            data['累積日照'].append(nan_float(row.find_all('td')[9].text))
            row_count += 1
    return pd.DataFrame(data)

def get_data(sql_db_path):
    for i, station_name in enumerate(station_id_table.keys()):
        this_df = get_weather_observation(station_name)
        if i == 0:
            observation_df = this_df
        else:
            observation_df = pd.concat([observation_df, this_df], axis=0, ignore_index=True)
    
    ##print(observation_df)
    
    conn = sqlite3.connect(sql_db_path)
    cursor = conn.cursor()

    sql_command = (
        'CREATE TABLE IF NOT EXISTS observation('
        'station VARCHAR(5), '
        'obs_time DATETIME, '
        'temperature FLOAT, '
        'weather_condition VARCHAR(10), '
        'wind_direction CHAR(5), '
        'wind_speed FLOAT, '
        'gust_wind FLOAT, '
        'relative humidity INT, '
        'sea_level_pressure FLOAT, '
        'rain_fall FLOAT, '
        'sun_light FLOAT'
        ');'
    )
    cursor.execute(sql_command)
    conn.commit()

    cursor.execute(f'SELECT station, obs_time FROM observation ORDER by obs_time DESC LIMIT {len(observation_df)}')
    recorded_data = cursor.fetchall()

    for i in range(len(observation_df)):
        this_list = list(observation_df.loc[i])
        if not (this_list[0], this_list[1]) in recorded_data:
            sql_command = (
                "INSERT INTO observation VALUES("
                f"'{this_list[0]}', "
                f"'{this_list[1]}', "
                f"{this_list[2]}, "
                f"'{this_list[3]}', "
                f"'{this_list[4]}', "
                f"{this_list[5]}, "
                f"{this_list[6]}, "
                f"{this_list[7]}, "
                f"{this_list[8]}, "
                f"{this_list[9]}, "
                f"{this_list[10]}"
                ");"
            )
            conn.execute(sql_command)
    conn.commit()
    cursor.close()
    conn.close()
