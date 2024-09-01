from model_management.Predict import evaluation

test_data_path = '../../historical/data/'

def evaluate(data_path):
    eval_df = evaluation(data_path)
    return eval_df

if __name__ == '__main__':
    data_path = ''
    eval_df = evaluate(test_data_path)
    print(eval_df)