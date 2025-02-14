#!/bin/bash

# 1. 清除旧的日志文件
if [ -f "uvicorn.log" ]; then
  echo "删除 uvicorn.log 文件"
  rm -f uvicorn.log
fi

if [ -f "server.log" ]; then
  echo "删除 server.log 文件"
  rm -f server.log
fi

# 2. 查找并停止已运行的 uvicorn 进程
echo "查找并停止已运行的 uvicorn 进程..."
# 通过 ps 命令和 grep 找到与 uvicorn app:app 相关的进程
PIDS=$(ps -aux | grep 'uvicorn app:app' | grep -v 'grep' | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "停止进程: $PIDS"
  kill -9 $PIDS
fi

# 3. 启动新的 uvicorn 进程
echo "启动新的 uvicorn 进程..."
nohup uvicorn app:app --log-level info --port 8123 --host 127.0.0.1 > uvicorn.log 2>&1 &

echo "uvicorn 启动成功，日志输出到 uvicorn.log"