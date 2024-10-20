from data_integration.integrating_power_data import get_oneday_power_data
import datetime

def get_power_generation_at_peak(sql_db):
    # 從 SQL 資料庫提取當天用電尖峰時的資料，儀表板要用
    now = datetime.datetime.now()
    today = now.date()
    if now.hour < 6:
        today -= datetime.timedelta(days=1)

    df = get_oneday_power_data(sql_db, today, solar_energy_day_only=False)
    result = {
        '尖峰負載': df['尖峰負載'].iloc[0]/10,
        '風力': df['風力發電'].iloc[0],
        '太陽能': df['太陽能發電'].iloc[0],
        '尖峰時間': str(df['時間'].iloc[0]),
    }
    return result