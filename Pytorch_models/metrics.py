import numpy as np

class Tensor_Metrics():
    def mae(y_truth, y_pred):
        return (y_truth-y_pred).abs().mean()

    def mse(y_truth, y_pred):
        return ((y_truth-y_pred)**2).mean()

    def r2(y_truth, y_pred):
        return 1- ((y_truth-y_pred)**2).mean() / y_truth.var()
    
class Array_Metrics():
    def mae(y_truth, y_pred):
        return np.mean(np.abs(y_truth.reshape(-1)-y_pred.reshape(-1)))

    def mse(y_truth, y_pred):
        return np.mean((y_truth.reshape(-1)-y_pred.reshape(-1))**2)

    def r2(y_truth, y_pred):
        return 1- np.mean((y_truth.reshape(-1)-y_pred.reshape(-1))**2) / np.var(y_truth)