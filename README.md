直接运行：

```
uvicorn app2:app --log-level info --port 8123 --host 127.0.0.1 > uvicorn.log 2>&1
```


docker里运行：
```shell
docker build -t path_planning .
docker run -d -p 8123:8123 --name path_planning_container -v /data/ImageData:/data/ImageData path_planning
```

停止运行方法：
```shell
docker stop path_planning_container && docker rm path_planning_container
```