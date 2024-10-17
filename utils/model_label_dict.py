# 這個檔案負責指定 model_label 與 model class 的對應關係

from sklearn.svm import SVR, NuSVR, SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

model_class_dict = {
    # classifier 的 label 與模型對應表
    'classifier':{
        'RandomForest': RandomForestClassifier,
        'XGBoost': XGBClassifier,
        'LightGBM': LGBMClassifier,
        'SVC': SVC,
        'NuSVC': NuSVC,
        'LogisticRegression': LogisticRegression,
    },
    # regressor 的 label 與模型對應表
    'regressor':{
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'XGBoost': XGBRegressor,
        'LightGBM': LGBMRegressor,
        'SVR': SVR,
        'NuSVR': NuSVR,
    }
}