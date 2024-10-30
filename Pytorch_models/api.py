import joblib
import numpy as np
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from .metrics import Tensor_Metrics
mae = Tensor_Metrics.mae

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class Best_Parameter_Saver():
    '''在 Pytorch 模型訓練過程中紀錄最佳模型參數，以訓練過程中的某個 metric 為判斷標準
    Arg:
        mode(str): 可以是'min', 'avg_min', 'max', 'avg_max'，設定以被評估 metric 的最小值、平均最小值、最大值或平均最大值為基準紀錄模型參數
        mv_length(int, optional): 設定被評估 metric 的移動平均數的平均窗口為幾個 epoch ，預設為 10
    '''
    def __init__(self, mode, mv_length=10):
        self.mode = mode
        if mode in ['min', 'avg_min']:
            self.best_indicator = 1e20
        if mode in ['max', 'avg_max']:
            self.best_indicator = -1e20
        self.indicators = []
        self.best_epoch = 0
        self.mv_length = mv_length
            
    def save_params(self, model):
        self.best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.best_epoch = len(self.indicators) - 1

    def load_params(self, model):
        model.load_state_dict(self.best_model_wts)
        return model
    
    def get_moving_average(self):
        if len(self.indicators) >= self.mv_length:
            return np.mean(self.indicators[-self.mv_length::])
        return self.best_indicator
    
    def update_func(self, indicator, model):
        self.best_indicator = indicator
        self.save_params(model)
        
    def update(self, model, indicator):
        self.indicators.append(indicator)
        this_indicator = indicator
        if 'avg' in self.mode:
            this_indicator = self.get_moving_average()

        if 'min' in self.mode and this_indicator < self.best_indicator:
            self.update_func(this_indicator, model)
        if 'max' in self.mode and this_indicator > self.best_indicator:
            self.update_func(this_indicator, model)


class Model_API():
    '''
    這個 class 包裝 Pytorch 模型的訓練、推論與參數存取功能
    輸出入資料形式都是 numpy array，class 會自動將輸入資料轉換成 pytorch tensor
    =====================================================================================
    有動用到的 module 與 class 如下
    import numpy as np
    from tqdm import tqdm
    import os
    import joblib
    import json
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    SimpleNN
    Best_Parameter_Saver
    ======================================================================================
    '''
    
    def __init__(self, model, L2_factor, linear_transform=True, classifier=False):
        '''
        Arg:
            model (Pytorch model instance): 核心 Pytorch 模型
            L2_factor (float): 訓練時的 L2 正則化參數
            linear_transform (bool, optional): 是否在DL模型之後整合一個線性轉換模型，預設為 True
            classifier (bool, optional): 是否為分類模型，若為 False 則為回歸模型，預設為 False
        Return:
            A Model_API instance
        '''
        self.model = model
        self.L2_factor = L2_factor
        self.linear_transform = linear_transform and not classifier
        self.classifier = classifier
        self.loss = self._loss_function_assignment()
    
    def _loss_function_assignment(self):
        if self.classifier:
            return nn.BCELoss() if self.model.params['output_f'] == 1 else nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _standard_scaler_fit(self, X, Y):
        self.scalerX = StandardScaler().fit(X)
        if not self.classifier:
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
            self.scalerY = StandardScaler().fit(Y)
     
    def data_preprocess(self, X, Y, batch_size, shuffle, standard_scale=True):
        if not self.classifier and len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        if standard_scale:
            X = self.scalerX.transform(X)
            if not self.classifier:
                Y = self.scalerY.transform(Y)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X.to(device), Y.to(device))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
    
    def _train_epoch(self, optimizer, train_loader, loss_function, metrics):
        self.model.train()
        total_loss, total_samples = 0, 0
        total_metrics = {k: 0 for k in metrics.keys()}
        # Loop over batches
        for this_X, this_Y in train_loader:
            this_batch_size = this_X.size(0)
            this_Y = this_Y.squeeze()
            if this_batch_size > 2:
                # Propogation and Weight adjustment
                optimizer.zero_grad()
                y_pred = self.model(this_X).squeeze()
                loss_train = loss_function(y_pred, this_Y)
                loss_train.backward()
                optimizer.step()
                # Record summation of loss and metrics, and number of samples
                total_loss += loss_train * this_batch_size
                for k, func in metrics.items():
                    total_metrics[k] += func(this_Y, y_pred) * this_batch_size
                total_samples += this_batch_size

        # Record loss and metrics for training set
        self.history['Train_loss'].append((total_loss / total_samples).item())
        for k in metrics.keys():
            self.history[f'Train_{k}'].append((total_metrics[k] / total_samples).item())

    def _validation_epoch(self, val_loader, loss_function, metrics):
        self.model.eval()
        total_loss, total_samples = 0, 0
        total_metrics = {k: 0 for k in metrics.keys()}
        # Calculate and Record loss and metrics for validation set
        for this_X, this_Y in val_loader:
            this_Y = this_Y.squeeze()
            with torch.no_grad():
                y_pred = self.model(this_X).squeeze()
            loss_val = loss_function(y_pred, this_Y)
            
            this_batch_size = this_X.size(0)
            total_loss += loss_val * this_batch_size
            for k, func in metrics.items():
                total_metrics[k] += func(this_Y, y_pred) * this_batch_size
            total_samples += this_batch_size

        self.history['Val_loss'].append((total_loss / total_samples).item())
        for k in metrics.keys():
            self.history[f'Val_{k}'].append((total_metrics[k] / total_samples).item())

    def _adjust_learning_rate(self, scheduler, scheduler_ref):
        # Adjust learning rate and save best weights if certain criterias have been satisfied
        ref = self.history[scheduler_ref][-1]
        scheduler.step(ref)
        self.best_parameter_saver.update(self.model, ref)

    def _early_stop_checkpoint(self, optimizer, verbose):
        # 檢查 lr 是否有改變，有改變的話印出提示訊息
        # 若 lr < min_lr 則結束訓練
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < self.min_lr:
            if verbose == 1:
                print(f'Learning rate is very small, terimnate training.')
            return True
        if not current_lr == self.previous_lr and verbose==1:
            print(f"Learning rate changed from {self.previous_lr} to {current_lr}")
        self.previous_lr = current_lr
        return False
    
    def _train_linear_transform_model(self, X_train, Y_train):
        if hasattr(self, 'scalerX'):
            test_X = torch.tensor(self.scalerX.transform(X_train), dtype=torch.float32)
        else:
            test_X = torch.tensor(X_train, dtype=torch.float32)
        self.Y_t = Y_train
        self.Y_p = self.model.to('cpu')(test_X).detach().numpy()
        if hasattr(self, 'scalerY'):
            self.Y_p = self.scalerY.inverse_transform(self.Y_p)

        self.LinearModel = LinearRegression()
        _ = self.LinearModel.fit(self.Y_p.reshape(-1, 1), self.Y_t.reshape(-1))

    def fit(
            self, X_train, Y_train, X_val=None, Y_val=None,
            standard_scale = True,
            loss_function = 'auto',
            metrics = {'MAE': mae},
            best_parametet_saver_mode='avg_min',
            best_parametet_saver_mv_length=10,
            scheduler_ref = 'Val_MAE',
            scheduler_mode = 'min',
            batch_size=16,
            n_epoch=1000,
            L2_factor=None,
            first_lr=1e-3,
            min_lr=1e-6,
            verbose=0
            ):
        '''訓練模型
        Args:
            X_train, Y_train (numpy array): 訓練集
            X_val, Y_val (numpy array, optional): 測試集
            standard_scale (bool, optional): 是否對 X 與 Y 進行 standard scaling，預設為 True
            loss_function (optional): Pytorch loss function or 'auto', automatically select a loss function base on model mode if 'auto' is assigned.
            metrics (dict, optional): 同時計算哪些 metrics，字典的 keys 為字串，做為 metric 的名字，values 則為計算 metric 的函數。預設為計算 MAE。
            best_parametet_saver_mode (str, optional): 參照 class Best_Parameter_Saver 裡的 mode 參數，預設為 'avg_min'
            best_parametet_saver_mv_length (int, optional): 參照 class Best_Parameter_Saver 裡的 mv_length 參數，預設為 10
            scheduler_ref (str, optional): lr_scheduler 參考的參數，預設為 'Val_MAE'，代表依據 Validation set 的 MAE 來調整 lr。
            scheduler_mode (str, optional): lr_scheduler 追求 ref 最大化 (max) 或最小化 (min)，預設為 'min'。
            batch_size (int, optional): 訓練時的 batch_size, 預設為 16
            n_epoch (int, optional): 最大訓練 epoch 數，預設為 1000
            L2_factor (float, optional): L2 正則化參數，若為 None 則引用 instance 本身的 L2_factor
            first_lr (float, optional): 起始 learning rate，預設為 1e-3
            verbose (int, optional): 訓練時印出的資訊詳細程度，可為 0 (資訊少) 或 1 (資訊多)
        '''
        # 初始化
        loss_function = self.loss if loss_function == 'auto' else loss_function
        L2_factor = self.L2_factor if L2_factor is None else L2_factor
        self.best_parameter_saver = Best_Parameter_Saver(mode=best_parametet_saver_mode, mv_length=best_parametet_saver_mv_length)
        validation = (not X_val is None) and (not Y_val is None)
        
        if standard_scale:
            self._standard_scaler_fit(X_train, Y_train)
        
        train_loader = self.data_preprocess(X_train, Y_train, batch_size, shuffle=True, standard_scale=standard_scale)
        if validation:
            val_loader = self.data_preprocess(X_val, Y_val, batch_size, shuffle=False, standard_scale=standard_scale)

        optimizer = optim.Adam(self.model.parameters(), lr=first_lr, weight_decay=L2_factor)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=10**(-0.5), patience=20)

        self.history = {'Train_loss':[]}
        for key in metrics.keys():
            self.history[f'Train_{key}'] = []
        if validation:
            self.history['Val_loss'] = []
            for key in metrics.keys():
                self.history[f'Val_{key}'] = []

        self.previous_lr = first_lr
        self.min_lr = min_lr
        self.model.to(device) # 若有 GPU 則將 model 移到 GPU

        # 訓練
        iterator = range(n_epoch)
        for epoch in iterator:
            self._train_epoch(optimizer, train_loader, loss_function, metrics)
            if validation:
                self._validation_epoch(val_loader, loss_function, metrics)
            self._adjust_learning_rate(scheduler, scheduler_ref)
            
            if verbose==1:
                # Print out informations of this epoch
                info_list = [f'Epoch: {epoch}'] + [f'{key}: {value[-1]:.4f}' for key, value in self.history.items()]
                info_string = ', '.join(info_list)
                print(info_string)
              
            if self._early_stop_checkpoint(optimizer, verbose):
                break

        # 將訓練過程暫存的最佳參數讀回來 
        self.model = self.best_parameter_saver.load_params(self.model)

        # 線性轉換
        if self.linear_transform:
            self._train_linear_transform_model(X_train, Y_train)
            
    # 進行推論
    def predict(self, X, linear_transform=True):
        if hasattr(self, 'scalerX'):
            X = self.scalerX.transform(X)
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            Y = self.model.to('cpu')(X).detach().numpy()
        if hasattr(self, 'scalerY'):
            Y = self.scalerY.inverse_transform(Y)
        if self.linear_transform and linear_transform:
            Y = self.LinearModel.predict(Y.reshape(-1,1))
        if self.classifier:
            Y[np.where(Y<0.5)] = 0
            Y[np.where(Y>=0.5)] = 1
        return np.squeeze(Y)

    # 模型參數存檔
    def save_weight(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        torch.save(self.model.state_dict(), file_path + 'FCN.pt')
        if hasattr(self, 'scalerX'):
            joblib.dump(self.scalerX, file_path + 'scalerX.model')
        if hasattr(self, 'scalerY'):
            joblib.dump(self.scalerY, file_path + 'scalerY.model')
        if self.linear_transform:
            joblib.dump(self.LinearModel, file_path + 'LinearModel.model')

        model_params_dict = self.model.params
        model_params_dict['linear_transform'] = self.linear_transform

        with open(file_path + 'hyper_parameters.json', 'w') as f:
            _ = json.dump(model_params_dict, f)

    # 從檔案讀取模型參數
    def load_weight(self, file_path):
        self.model.load_state_dict(torch.load(file_path + 'FCN.pt', map_location=device))
        if os.path.exists(file_path + 'scalerX.model'):
            self.scalerX = joblib.load(file_path + 'scalerX.model')
        if os.path.exists(file_path + 'scalerY.model'):
            self.scalerY = joblib.load(file_path + 'scalerY.model')
        if self.linear_transform:
            self.LinearModel = joblib.load(file_path + 'LinearModel.model')