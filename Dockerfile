# 基础镜像
FROM python:3.12.7-slim-bookworm

# 设置工作目录为 /app（容器内自动创建）
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制所有当前目录文件到容器的工作目录（/app）
COPY . .

# 暴露端口
EXPOSE ${APP_PORT:-8123}

# 启动命令（注意路径！）
CMD uvicorn app:app --host 0.0.0.0 --port ${APP_PORT:-8123} --log-level info