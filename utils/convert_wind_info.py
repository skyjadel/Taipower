import numpy as np
from typing import List
from pandas import DataFrame

def polar_to_cartesian_coord(df: DataFrame, input_col_names: List[str], output_col_names: List[str]) -> DataFrame:
    '''將風速風向轉成東西風與南北風，並刪除風向欄位
    Arg:
        df (DataFrame): 要轉換的資料表，需包含風速與風向資料，風向資料需為 360 度表示法，正北為 0，正東為 90
        input_col_names (List[str]): 風速與風向的欄位名，順序為 [風速, 風向]
        output_col_names (List[str]): 東西風與南北風的欄位名，順序為 [東西風, 南北風]
    '''
    wind_speed = list(df[input_col_names[0]])
    wind_direction = list(df[input_col_names[1]] / 180 * np.pi)
    NS_wind = np.abs(wind_speed * np.cos(wind_direction))
    EW_wind = np.abs(wind_speed * np.sin(wind_direction))
    df[output_col_names[0]] = EW_wind
    df[output_col_names[1]] = NS_wind
    df.drop([input_col_names[1]], axis=1, inplace=True)
    return df