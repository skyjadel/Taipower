from model_management.Train import main_train

params_dict = {
    'meta_path': '../../trained_model_parameters/model_meta_2024-09-10/', 
    'data_path': '../../historical/data/', 
    'test_size': 0.001,
    'test_last_fold': False,
    'apply_night_peak': False,
    'start_date': '2023-08-01',
    'end_date': '2200-12-31'
}
train_model_main_path = '../../trained_model_parameters/models_tobe_evaluated/'

def main(params_dict, train_model_main_path, preserved_days=0):
    main_train(params=params_dict, train_model_main_path=train_model_main_path, preserved_days=preserved_days,
            apply_night_peak=False, remove_night_peak_samples=True)
    main_train(params=params_dict, train_model_main_path=train_model_main_path, preserved_days=preserved_days,
            apply_night_peak=True, remove_night_peak_samples=True)
    main_train(params=params_dict, train_model_main_path=train_model_main_path, preserved_days=preserved_days,
            apply_night_peak=False, remove_night_peak_samples=False)
    
if __name__ == '__main__':
    main(params_dict, train_model_main_path, preserved_days=10)