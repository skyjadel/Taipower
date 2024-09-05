import os
import pandas as pd
import datetime
import numpy as np

from model_management.Predict import main_predict

test_data_path = '../../historical/data/'
test_model_dir = '../../trained_model_parameters/models_tobe_evaluated/'
test_current_model_path = '../../trained_model_parameters/latest_model/'

def evalute_models(data_path, model_dir, current_model_path, evaluation_tolerence_days=0):
    avoid_training_set = (evaluation_tolerence_days <= 0)
    power_features = ['風力', '太陽能', '尖峰負載']

    # 決定 model 列表
    model_list = os.listdir(model_dir)
    if 'evaluations.csv' in model_list:
        model_list.remove('evaluations.csv')
    model_list.append('Current')

    power_df = pd.read_csv(f'{data_path}power/power_generation_data.csv')
    power_df['日期'] = pd.to_datetime(power_df['日期'])
    power_df['尖峰負載'] /= 10
    power_df.rename({'太陽能發電': '太陽能', '風力發電':'風力'}, axis=1, inplace=True)

    forecast_df = pd.read_csv(f'{data_path}weather/finalized/weather_forecast.csv')
    forecast_df['日期'] = pd.to_datetime(forecast_df['日期'])
    forecast_last_given_date = max(forecast_df['日期']).to_pydatetime()

    predict_day_dict = {}

    for model in model_list:
        if model == 'Current':
            this_model_path = current_model_path
        else:
            this_model_path = f'{model_dir}{model}/'

        model_training_df = pd.read_csv(this_model_path + '太陽能/data.csv')
        model_training_df['日期'] = pd.to_datetime(model_training_df['日期'])
        this_model_enddate = max(model_training_df['日期'])

        predict_days = int((forecast_last_given_date - this_model_enddate)/datetime.timedelta(days=1)) + evaluation_tolerence_days
        predict_day_dict[model] = predict_days

    unique_pred_day_list = list(set([v for v in predict_day_dict.values()]))

    def evaluate_one_model(model, predict_days,
                           data_path=data_path,
                           avoid_training_set=avoid_training_set,
                           save_prediction=False):
        
        print(f'Evaluation of {model} ({predict_days} days)......', end=' ')

        eval_result_dict = {}

        if model == 'Current':
            this_model_path = current_model_path
        else:
            this_model_path = f'{model_dir}{model}/'

        power_pred_df = main_predict(data_path=data_path,
                                     model_path=this_model_path,
                                     predict_days=predict_days, 
                                     avoid_training_set=avoid_training_set,
                                     save_prediction=save_prediction)
        power_pred_df['日期'] = pd.to_datetime(power_pred_df['日期'])

        power_obs_pred_df = pd.merge(power_pred_df, power_df, on='日期', suffixes=['_預測', '_觀測'])
        
        for y_f in power_features:
            YP = power_obs_pred_df[f'{y_f}_預測']
            YT = power_obs_pred_df[f'{y_f}_觀測']
            MAE = np.mean(np.abs(YP - YT))
            eval_result_dict[f'{y_f}_MAE'] = MAE
        eval_result_dict['Model'] = model 
        eval_result_dict['Samples'] = len(power_obs_pred_df)
        print('is done.')

        return pd.DataFrame(eval_result_dict, index=[0])
    
    result_df_list = []
    for pred_days in unique_pred_day_list:
        for model in model_list:
            if predict_day_dict[model] >= pred_days:
                this_df = evaluate_one_model(model, predict_days=pred_days)
                result_df_list.append(this_df)
        
    total_eval_df = pd.concat(result_df_list, axis=0, ignore_index=True).reset_index(drop=True)
    total_eval_df.to_csv(f'{model_dir}evaluations.csv', index=False, encoding='utf-8-sig')
    return total_eval_df

if __name__ == '__main__':
    evalute_models(test_data_path, test_model_dir, test_current_model_path, evaluation_tolerence_days=5)
