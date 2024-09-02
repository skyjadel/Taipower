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
    def __init__(self, mode, mv_length=10):
        self.mode = mode
        if mode in ['min', 'avg_min']:
            self.best_indicator = 1e20
        elif mode in ['max', 'avg_max']:
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
        
    def update(self, model, indicator):
        self.indicators.append(indicator)
        if self.mode == 'min':
            if indicator < self.best_indicator:
                self.best_indicator = indicator
                self.save_params(model)
        elif self.mode == 'max':
            if indicator > self.best_indicator:
                self.best_indicator = indicator
                self.save_params(model)
        elif 'avg' in self.mode:
            if len(self.indicators) >= self.mv_length:
                moving_avg = np.mean(self.indicators[-self.mv_length::])
                if self.mode == 'avg_min':
                    if moving_avg < self.best_indicator:
                        self.best_indicator = moving_avg
                        self.save_params(model)
                elif self.mode == 'avg_max':
                    if moving_avg > self.best_indicator:
                        self.best_indicator = moving_avg
                        self.save_params(model)


class Model_API():
    # 這個 class 的輸出入資料形式都是 numpy array，class 會自動將輸入資料轉換成 pytorch tensor
    # =====================================================================================
    # 有動用到的 module 與 class 如下
    # import numpy as np
    # from tqdm import tqdm
    # import os
    # import joblib
    # import json
    # import torch
    # import torch.nn as nn
    # from torch.utils.data import DataLoader, TensorDataset
    # import torch.optim as optim
    # from sklearn.preprocessing import StandardScaler
    # SimpleNN
    # Best_Parameter_Saver
    # ======================================================================================
    
    def __init__(self, model, L2_factor, linear_transform=True, classifer=False):
        self.model = model
        self.L2_factor = L2_factor
        self.linear_transform = linear_transform
        self.classifer = classifer
        if classifer:
            if self.model.params['output_f'] == 1:
                self.loss = nn.BCELoss()
            else:
                self.loss = nn.CrossEntropyLoss()
            self.linear_transform = False
        else:
            self.loss = nn.MSELoss()
    
    def standard_scaler_fit(self, X, Y):
        self.scalerX = StandardScaler().fit(X)
        if not self.classifer:
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
            self.scalerY = StandardScaler().fit(Y)

        
    def data_preprocess(self, X, Y, batch_size, shuffle, standard_scale=True):
        if not self.classifer:
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
        if standard_scale:
            X = self.scalerX.transform(X)
            if not self.classifer:
                Y = self.scalerY.transform(Y)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X.to(device), Y.to(device))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    # 訓練模型
    def fit(self, X_train, Y_train, X_val=None, Y_val=None,
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
            verbose=0):
        
        if loss_function == 'auto':
            loss_function = self.loss
        
        if L2_factor is None:
            L2_factor = self.L2_factor

        self.best_parameter_saver = Best_Parameter_Saver(mode=best_parametet_saver_mode, mv_length=best_parametet_saver_mv_length)
        validation = True
        if X_val is None or Y_val is None:
            validation = False
        
        if standard_scale:
            self.standard_scaler_fit(X_train, Y_train)
        train_loader = self.data_preprocess(X_train, Y_train, batch_size, shuffle=True, standard_scale=standard_scale)
        if validation:
            val_loader = self.data_preprocess(X_val, Y_val, batch_size, shuffle=False, standard_scale=standard_scale)

        optimizer = optim.Adam(self.model.parameters(), lr=first_lr, weight_decay=L2_factor)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=10**(-0.5), patience=20)
        criterion = loss_function

        self.history = {'Train_loss':[]}
        for key in metrics.keys():
            self.history[f'Train_{key}'] = []
        if validation:
            self.history['Val_loss'] = []
            for key in metrics.keys():
                self.history[f'Val_{key}'] = []

        previous_lr = first_lr
        self.model.to(device)

        iterator = range(n_epoch)
        for epoch in iterator:
            self.model.train()
            total_loss, total_samples = 0, 0
            total_metrics = {k: 0 for k in metrics.keys()}
            for this_X, this_Y in train_loader:
                this_batch_size = this_X.size(0)
                this_Y = this_Y.squeeze()
                if this_batch_size > 2:
                    optimizer.zero_grad()
                    y_pred = self.model(this_X).squeeze()
                    loss_train = criterion(y_pred, this_Y)
                    loss_train.backward()
                    optimizer.step()
                    #print(loss_train)
                    total_loss += loss_train * this_batch_size
                    for k, func in metrics.items():
                        total_metrics[k] += func(this_Y, y_pred) * this_batch_size
                    total_samples += this_batch_size

            self.history['Train_loss'].append((total_loss / total_samples).item())
            for k in metrics.keys():
                self.history[f'Train_{k}'].append((total_metrics[k] / total_samples).item())

            if validation:
                self.model.eval()
                total_loss, total_samples = 0, 0
                total_metrics = {k: 0 for k in metrics.keys()}
                for this_X, this_Y in val_loader:
                    this_Y = this_Y.squeeze()
                    with torch.no_grad():
                        y_pred = self.model(this_X).squeeze()
                    loss_val = criterion(y_pred, this_Y)
                    
                    this_batch_size = this_X.size(0)
                    total_loss += loss_val * this_batch_size
                    for k, func in metrics.items():
                        total_metrics[k] += func(this_Y, y_pred) * this_batch_size
                    total_samples += this_batch_size

                self.history['Val_loss'].append((total_loss / total_samples).item())
                for k in metrics.keys():
                    self.history[f'Val_{k}'].append((total_metrics[k] / total_samples).item())
                
                ref = self.history[scheduler_ref][-1]
                scheduler.step(ref)
                self.best_parameter_saver.update(self.model, ref)
            else:
                scheduler.step(ref)
                self.best_parameter_saver.update(self.model, ref)
            
            if verbose==1:
                info_string = f'Epoch: {epoch}, '
                for key, value in self.history.items():
                    info_string += f'{key}: {value[-1]:.4f}, '
                print(info_string)
                
            current_lr = optimizer.param_groups[0]['lr']
            if not current_lr == previous_lr:
                if current_lr < 1e-6:
                    if verbose == 1:
                        print(f'Learning rate is very small, terimnate training.')
                    break
                if verbose == 1:
                    print(f"Learning rate changed from {previous_lr} to {current_lr}")
                previous_lr = current_lr
                
        self.model = self.best_parameter_saver.load_params(self.model)
        if self.linear_transform:
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
        if self.classifer:
            Y[np.where(Y<0.5)] = 0
            Y[np.where(Y>=0.5)] = 1
        Y = np.squeeze(Y)
        return Y

    # 儲存模型參數
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

    # 讀取模型參數
    def load_weight(self, file_path):
        self.model.load_state_dict(torch.load(file_path + 'FCN.pt'))
        if os.path.exists(file_path + 'scalerX.model'):
            self.scalerX = joblib.load(file_path + 'scalerX.model')
        if os.path.exists(file_path + 'scalerY.model'):
            self.scalerY = joblib.load(file_path + 'scalerY.model')
        if self.linear_transform:
            self.LinearModel = joblib.load(file_path + 'LinearModel.model')