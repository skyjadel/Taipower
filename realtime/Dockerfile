FROM apache/airflow:2.9.2-python3.9

# 設定環境變數
ENV AIRFLOW_HOME=/opt/airflow

# 設定Workdir
WORKDIR $AIRFLOW_HOME

# 安裝需要的套件
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 安裝 Chrome 瀏覽器
USER root
RUN apt-get clean && apt-get update && apt-get install -y wget gnupg2 \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update && apt-get install -y google-chrome-stable

# # 安裝 ChromeDriver
# RUN CHROMEDRIVER_VERSION=`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE` \
#     && wget -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip \
#     && unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/ \
#     && rm /tmp/chromedriver.zip

# 切換回 airflow 使用者
USER airflow

# 複製你的專案文件到容器
COPY src .

# 設定 Airflow 使用 PostgreSQL
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor