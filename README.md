# 以氣象資料預測台電尖峰負載與綠能發電量


## 專案概述
這個專案想要以氣象資料，比如說每天的最高氣溫，來預測台電的尖峰負載。
同時台電的發電結構當中，風力與太陽能的比例節節上升，已經超越核能，然而風力和太陽能是字面上意義的靠天吃飯，所以這個專案也想要藉由風速與日照等觀測資料，來預測風力與太陽能的每天發電量。

不過既然要預測，我們總是希望能從前一天的氣象預報來預測第二天的電力需求與綠能產出，然而我並沒有找到中央氣象署公開的歷史預測資料。
因此只好從專案開啟那一天開始使用 airflow 自動收集中央氣象署官網上的預報資料，希望在累積足夠的資料之後，能夠了解從預報轉換到觀測資料的方式。

本專案接下來還想嘗試利用開源 LLM，利用 RAG 技術串接預測模型，讓 LLM 可以回答以上關於電力預測的問題。

## 專案結構
./realtime
  這個資料夾本身是一個 docker 專案，所以其中的 Dockerfile, docker-compose.yml, 與 requirements.txt 都是建立 docker container 所需的檔案
  這個 container 的目的是執行兩個 airflow 任務，分別定期自動去爬取實時氣象觀測資料，實時氣象預報 (氣象資料每六小時爬取一次)，以及實時台電各機組發電量 (電力資料每30分鐘爬取一次)
  ./realtime/src
    當中準備了爬取中央氣象署與台灣電力公司網站的爬蟲程式
  ./realtime/dags
    當中準備了 airflow 所需的 dag 檔案，定義了兩個 airflow 任務
  ./realtime/realtime_data
    存放一個 realtime.db 檔，它是一個 sqlite3 的資料庫檔案，存放前面所述爬蟲爬取的資料

./historical
  存放天氣與電力的歷史資料
  其中整合過的歷史天氣資料路徑為 ./historical/data/weather/finalized/big_table.csv
  整合過的歷史電力資料路徑為 ./historical/data/power/power_deneration_data.csv
  ./historical/src
    一些整理下載後的歷史資料的程式

./utils
  存放我自己寫的輔助資料分析的模組

./EDA.ipynb                         EDA 用 jupyter notebook
./Test_realtime_airflow.ipynb       印出 airflow 與爬蟲抓取的最新資料，以便以肉眼檢查資料正確性
./Power_prediction.ipynb            這個筆記本包含從天氣觀測資料預測電力資料的特徵工程，建模與評估模型部分

## 安裝與使用方法
提供安裝和使用你的專案的指示。包括所需的相依套件、如何設置環境、安裝步驟和執行專案的指令。

## 資料收集
如果你的專案需要特定的資料集，這裡可以提供一些關於如何獲取、處理或下載資料的指示。

## 資料分析流程
這一部分描述你的資料分析流程，包括你使用的方法、模型或演算法。你可以提供程式碼片段或流程圖來幫助讀者理解你的分析過程。

## 結果展示
展示和解釋你的資料分析結果。可以包括圖表、視覺化效果或統計數據，並提供解釋和洞察。

