FROM apache/airflow:2.9.2-python3.9

# 設定環境變數
ENV AIRFLOW_HOME=/opt/airflow

# 設定Workdir
WORKDIR $AIRFLOW_HOME

# 安裝需要的套件
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio
RUN pip3 install selenium
RUN pip3 install webdriver-manager

# 安裝 Chrome 瀏覽器
USER root
RUN apt-get clean && apt-get update && apt-get install -y wget gnupg2 unzip libgomp1\
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update && apt-get install -y google-chrome-stable

# 確保安裝 Chrome 115 以上的版本
RUN google-chrome --version

# 切換回 airflow 使用者
USER airflow

# 複製專案文件到容器
COPY src .

# 複製 packages 到容器
COPY ../Pytorch_models .
COPY ../crawler .
COPY ../data_integration .
COPY ../model_management .
COPY ../utils .
