import torch.nn as nn
import torch

class SimpleNN(nn.Module):
    def __init__(self, input_f, output_f, feature_counts,
                  dropout_factor=0,
                  positive_define=False):
        super(SimpleNN, self).__init__()
        self.params = {
            'input_f': input_f,
            'output_f': output_f,
            'feature_counts': feature_counts,
            'dropout_factor': dropout_factor,
            'positive_define': positive_define
        }
        self.feature_counts = feature_counts
        self.dropout_factor = dropout_factor
        self.positive_define = positive_define
        self.BN0 = nn.BatchNorm1d(input_f)
        self.blocks = nn.ModuleList()
        for i, f in enumerate(feature_counts):
            out_f = f
            if i == 0:
                in_f = input_f
            else:
                in_f = feature_counts[i-1]
            self.blocks.append(self.fnn_block(in_f, out_f))
        self.output_layer_A = nn.Linear(feature_counts[-1], output_f)
        self.output_layer_B = nn.Linear(output_f, output_f)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    
    def fnn_block(self, in_f, out_f, dropout=True, activation=True):
        layers = []
        layers.append(nn.Linear(in_f, out_f))
        layers.append(nn.BatchNorm1d(out_f))
        if activation:
            layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(p=self.dropout_factor))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.BN0(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer_A(x)
        output = self.output_layer_B(x)
        if self.positive_define:
            output = self.relu(output)
        return output
    
class SimpleNN_classifier(nn.Module):
    def __init__(self, input_f, output_f, feature_counts,
                  dropout_factor=0,
                  positive_define=False):
        super(SimpleNN_classifier, self).__init__()
        self.params = {
            'input_f': input_f,
            'output_f': output_f,
            'feature_counts': feature_counts,
            'dropout_factor': dropout_factor,
            'positive_define': positive_define
        }
        self.feature_counts = feature_counts
        self.dropout_factor = dropout_factor
        self.positive_define = positive_define
        self.BN0 = nn.BatchNorm1d(input_f)
        self.blocks = nn.ModuleList()
        for i, f in enumerate(feature_counts):
            out_f = f
            if i == 0:
                in_f = input_f
            else:
                in_f = feature_counts[i-1]
            self.blocks.append(self.fnn_block(in_f, out_f))
        self.output_layer_A = nn.Linear(feature_counts[-1], output_f)
        self.output_layer_B = nn.Linear(output_f, output_f)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    
    def fnn_block(self, in_f, out_f, dropout=True, activation=True):
        layers = []
        layers.append(nn.Linear(in_f, out_f))
        layers.append(nn.BatchNorm1d(out_f))
        if activation:
            layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(p=self.dropout_factor))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.BN0(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer_A(x)
        x = self.output_layer_B(x)
        output = self.sigmoid(x)
        
        return output
    
class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
        super(LSTM_model, self).__init__()
        self.params = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'output_size': output_size
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers, 
                             dropout=dropout, 
                             batch_first=True)
        
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.LSTM(x, (h0, c0))
        output = self.linear(lstm_out[:, -1, :])
        return output