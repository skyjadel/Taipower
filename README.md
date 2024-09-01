# 以氣象資料預測台電尖峰負載與綠能發電量


## 專案概述
這個專案想要以氣象資料，比如說每天的最高氣溫，來預測台電的尖峰負載。
同時台電的發電結構當中，風力與太陽能的比例節節上升，已經超越核能，然而風力和太陽能是字面上意義的靠天吃飯，所以這個專案也想要藉由風速與日照等觀測資料，來預測風力與太陽能的每天發電量。

不過既然要預測，我們總是希望能從前一天的氣象預報來預測第二天的電力需求與綠能產出，然而我並沒有找到中央氣象署公開的歷史預測資料。
因此只好從專案開啟那一天開始使用 airflow 自動收集中央氣象署官網上的預報資料，並在累積足夠的資料之後，建立預報轉換到觀測資料的模型。

如上所述，這個專案的資料預測部份分成兩個階段：
1. 從氣象預報資料預測氣象觀測資料。
2. 從氣象觀測資料以及其他包括假日與季節因素預測綠能與尖峰負載。

以上兩個部份的訓練與預測都採用了集成學習，使用的模型種類包括：  
Linear Regression  
Logistic Regression  
Random Forest  
XGBoost  
LightGBM  
SVM  
NuSVM  
Fully-Connected Neural Network  

本專案也利用 GPT4 API 與 langchain RAG 技術串接預測結果，讓 LLM 可以回答以上關於電力預測的問題。另外也使用 streamlit 套件製作一個儀表板，呈現最新預測以及過去預測的準確度。
這些部份的呈現可以在 <a href="http://ec2-54-206-30-159.ap-southeast-2.compute.amazonaws.com:8501/">這裡</a> 找到。

## 專案結構
./realtime/realtime_data  
存放一個 realtime.db 檔，它是一個 sqlite3 的資料庫檔案，存放前面所述爬蟲爬取的資料  
以及一個 peak.csv 檔，紀錄今天到目前為止用電尖峰的資料，以提供 dashboard.py 使用  
    
./historical
存放天氣與電力的歷史資料  
其中歷史天氣資料路徑為 ./historical/data/weather/finalized/big_table.csv  
歷史電力資料路徑為 ./historical/data/power/power_deneration_data.csv  
歷史預測資料為 ./historical/data/prediction/  

以下五個資料夾為我自己寫的套件  
./crawler  
負責將即時天氣預報、天氣觀測、電力資訊利用爬蟲抓取下來，並存到 realtime.db 資料庫中  
./data_integration  
負責將 sqlite3 database 裡的即時資料整合成歷史資料格式，並存到 ./historical 資料夾  
./model_management  
提供各機器學習/深度學習的集成學習整合 API，包括模型的訓練、呼叫與存取  
./Pytorch_models  
包含 PyTorch 深度學習模型的定義，以及 API 包裝，API功能包括模型創建、訓練、預測，以及參數的存取  
./utils  
其他需要的模組，以及一些程式需要的先備知識  

./airflow-docker  
存放建立 Docker 所需檔案。這個 Docker 的功能是利用 airflow 定期進行資料爬取、資料整合、模型預測、模型評估、模型訓練等任務  

./EDA.ipynb  
EDA 用 jupyter notebook  

./Test_realtime_airflow.ipynb  
印出 airflow 與爬蟲抓取的最新資料，以便以肉眼檢查資料正確性  

./Power_prediction.ipynb  
這個筆記本包含從天氣觀測資料預測電力資料的特徵工程，建模與評估模型部分  

./Forecast_to_Weather_obs_Hyperparameter.ipynb  
這個筆記本包含從氣象預報資料預測氣象觀測資料的特徵工程、建模、超參數調整、集成學習與評估模型部分  

./Weather_obs_to_Power_Hyperparameter.ipynb  
這個筆記本包含從天氣觀測資料預測電力資料的超參數調整、集成學習與評估模型部分  

./dashboard.py  
定義 streamlit 儀表板，就是我們在 <a href="http://ec2-54-206-30-159.ap-southeast-2.compute.amazonaws.com:8501/">這裡</a> 看到的  

./chatbot.py  
定義聊天機器人，利用模型預測結果與真實電力資訊回答電力相關問題。  

## 安裝與使用方法
### 系統需求
由於這個專案有使用一點點的深度學習，所以系統最好要有與 cuda 相容的顯示卡，VRAM 最好要在 6GB 以上
### 自動執行設定
這個專案的自動執行部分使用 airflow 與 Docker 進行，因此我們要先安裝 Docker  
如果你是使用 Windows 10 或 Windows 11，可以參考 <a href="https://medium.com/@weiberson/%E5%9C%A8win11%E5%AE%89%E8%A3%9Dwsl%E5%92%8Cdocker%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-6d50473b5e09">這篇</a> 安裝 WSL2 與 Docker  
接下來請準備一個乾淨的資料夾 (以下以 D:\Taipower\ 為例)，將專案在裡面複製一份  
然後打開 WSL2 的終端機，在命令列執行 cd /mnt/d/Taipower/airflow-docker

## 資料收集
如果你的專案需要特定的資料集，這裡可以提供一些關於如何獲取、處理或下載資料的指示。

## 資料分析流程
這一部分描述你的資料分析流程，包括你使用的方法、模型或演算法。你可以提供程式碼片段或流程圖來幫助讀者理解你的分析過程。

## 結果展示
展示和解釋你的資料分析結果。可以包括圖表、視覺化效果或統計數據，並提供解釋和洞察。

