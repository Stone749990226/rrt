# 基础镜像
FROM python:3.12.7-slim-bookworm

# 设置工作目录为 /app（容器内自动创建）
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 升级 pip 并安装依赖（使用清华镜像源）
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制所有当前目录文件到容器的工作目录（/app）
COPY . .

RUN mkdir -p /logs

# 启动命令（JSON 格式，避免信号问题）
CMD ["python", "main.py"]
