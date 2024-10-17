from model_management.Predict import main_predict

test_data_path = '../../historical/data/'
test_model_path = '../../trained_model_parameters/latest_model/'

def predict(data_path, model_path, predict_days=1):
    _ = main_predict(data_path=data_path, model_path=model_path, predict_days=predict_days, wind_speed_naive=True)

if __name__ == '__main__':
    data_path = ''
    predict(test_data_path, test_model_path, predict_days=15)